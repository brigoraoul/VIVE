import os
import logging
import torch
import re
from openai import OpenAI
import ollama
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, Conversation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from PersonalValueAgent.value_extraction_sources.i_value_extraction_source import IValueExtractionSource
from PersonalValueAgent.db_utils.database_utils import DAO

# templates for prompts, to be filled with message and value(s)
PROMPT_TEMPLATES = {
    "single_value": "Is the following personal value an underlying value for the following message? "
                    "Message: {} "
                    "Personal Value: {} "
                    "To answer the question, consider, whether the following sentence is a correct statement: The "
                    "author composed this message, because {} is important to him/her? "
                    "Answer only with 'yes' or 'no'!",
    "single_value_validation": "Is the following personal value an underlying value for the following message? "
                               "Message: {} "
                               "Personal Value: {} "
                               "Answer only with 'yes' or 'no'!",
    "single_label": "Which of the following personal values is the underlying value of the following message? "
                    "Message: {} "
                    "Personal Values: [{}] "
                    "To answer the question, consider, which one of the personal values is the most important to the "
                    "author of the message. "
                    "Answer only by stating the underlying values in this format [<value>]! Do not give any "
                    "additional information.",
    "single_prompt": "Which of the following personal values are underlying values for the following message?"
                     "Message: {}"
                     "Personal Values: {}"
                     "For each value consider, whether filling the blank in the following sentence results in a"
                     "correct statement: The author composed this message, because <value> is important to him/her?"
                     "Answer only by listing the underlying values in this format [<value1>,<value2>,...,<valueN>]",
    "question_answering": "The author composed this message, because {} is important to him/her?",
    "context": "You are an analyst from a humanitarian organisation. You have collected messages from "
               "social media groups, in which war refugees ask questions and get information. You are "
               "interested in the personal values that someone references.",
    "context_with_description": "You are an analyst from a humanitarian organisation. You have collected messages from "
                                "social media groups, in which war refugees ask questions and get information. You "
                                "have identified the following five personal values that people deem particularly "
                                "important in the context of fleeing the war in their country and settling in a "
                                "different country: "
                                "Personal Value 1: shelter "
                                "Keywords: housing "
                                "Description: Ukrainian refugees need a shelter/place to stay or to live when they "
                                "flee to other regions of Ukraine or to other/neighbouring countries. This is of "
                                "paramount importance. "
                                "Personal Value 2: mental health "
                                "Keywords: psychological support, psychological health "
                                "Description: They look for and offer psychological support because many people are "
                                "traumatized by war. "
                                "Personal Value 3: staying connected "
                                "Keywords: connectivity, mobile phone, Internet "
                                "Description: Good and inexpensive service provider (mobile phone, Internet) is very "
                                "important to stay in touch with their families/husbands. "
                                "Personal Value 4: disappointment in this city/country "
                                "Keywords: go back home "
                                "Description: From time to time we see messages when people get disappointed in a new "
                                "place and decide to go back to Ukraine even though it is unsafe over there. "
                                "Personal Value 5: help for refugees "
                                "Keywords: humanitarian aid "
                                "Description: Humanitarian and other kinds of help is needed for the people affected. "
                                "Looking at the collected messages from social media, you are interested in "
                                "understanding which of the personal values the author of a message references.",
    "context_with_question": "There is a dataset with messages from war refugees. You are an analyst from a "
                             " humanitarian organisation and you want to understand what personal values the author of "
                             " a message holds."
                             "Is the following personal value an underlying value for the following message? "
                             "Message: {} "
                             "Personal Value: {} "
                             "To answer the question, consider, whether filling the blank in the following sentence results in a"
                             " correct statement: The author composed this message, because {} is important to him/her? "
                             "Start your answer with 'Yes' or 'No'!",
}


def load_model(model_name, num_labels):
    """
    Load a pre-trained language model for text- / sequence-classification from huggingface.
    Args:
        model_name (String): Name of the model to be loaded from huggingface.
        num_labels (int): Number of labels for the classification task.
    Returns:
        model: The loaded model, for example "BertForSequenceClassification"
    """
    # get model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # save model to local file
    model_path = os.path.join(LLM.model_directory, model_name)
    model.save_pretrained(model_path)

    return model


def fine_tune(model, model_name, batch_size, num_epochs):
    """
    Fine-tune the loaded model for the classification task: Determining what values are referenced in a given text.
    Args:
        model: Language model from huggingface.
        model_name: Name of the model, needed to load suitable tokenizer.
    """
    messages = LLM.db_helper.get_messages()

    # tokenize messages and create dataloader to iterate over batches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_messages = tokenizer(messages, padding=True, truncation=True, return_tensors='pt')

    # Convert tokenized_messages to TensorDataset
    tensor_dataset = TensorDataset(tokenized_messages['input_ids'],
                                   tokenized_messages['attention_mask'])

    dataloader = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size)

    # set torch device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = num_epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch[0].to(device)  # input_ids
            attention_mask = batch[1].to(device)  # attention_mask

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # save model to local file
    model_path = os.path.join(LLM.model_directory, model_name)
    model.save_pretrained(model_path)


class LLM(IValueExtractionSource):
    """
    This class provides an api to a large language model (LLM) through which the LLM can be prompted and the response
    can be received. The class implements multiple prompt strategies that extract values via different prompts.
    The LLM is exposed on localhost via the inference server from LM Studio.

    Args:
        values (dict): Dictionary of value representations with value id as key.
        prompt_strategy (int): Indication of how LLM should be prompted
        history_on (bool): Flag to determine if chat history with LLM is maintained.

    Attributes:
        values (dict): Dictionary of value representations with value id as key.
        prompt_strategy (int): Indication of how LLM should be prompted
        history_on (bool): Flag to determine if chat history with LLM is maintained.
        client (OpenAI): OpenAI client instance.
        history (list): List to store chat history with LLM.
    """

    # helper that provides functions to access database
    db_helper = DAO()

    # local path to directory with word embedding model
    model_directory = "/Users/raoulbrigola/Documents/Documents/AIMaster/Thesis/MasterThesis/" \
                      "Value_Extraction/PersonalValueAgent/models"

    def __init__(self, values, prompt_strategy, history_on=False, simple_value_representation=False):
        self.values = values
        self.prompt_strategy = prompt_strategy
        self.history_on = history_on

        # llm studio models
        self.client = OpenAI(base_url="http://localhost:5002/v1", api_key="lm-studio")
        self.prompt_count = 0
        self.history = []

        # ollama model name
        self.ollama_model_name = "llama3"
        self.context = PROMPT_TEMPLATES["context_with_description"]

        simple_value_representation = True
        if simple_value_representation:
            self.context = PROMPT_TEMPLATES["context"]

        # huggingface models
        self.model_names = {
            "text_generation": "bert-base-uncased",
            "zero_shot_classification": "facebook/bart-large-mnli",
            "question_answering": "deepset/roberta-base-squad2",
            "conversation": "facebook/blenderbot-400M-distill"
        }

        self.num_labels = len(self.values)

        # load fine-tuned model from local path if it exists, otherwise load pre-trained model from huggingface
        model_path = os.path.join(LLM.model_directory, self.model_names["text_generation"])
        if not os.path.exists(model_path):
            model = load_model(self.model_names["text_generation"], self.num_labels)
            fine_tune(model, self.model_names["text_generation"], batch_size=8, num_epochs=1)

        self.model = BertForSequenceClassification.from_pretrained(model_path)

        # set torch device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def get_values_for_message(self, message, single_label):
        """
        Get underlying values for the input message.
        Args:
            message (str): Input message for which values should be extracted.
        Returns:
            list: List of extracted values.
            single_label: Indication, whether to predict a single value or a subset of values
        """
        llm_result = self.prompt(message, single_label)
        extracted_values = []
        extracted_values_as_vec = []  # vector representation with 0s (value not referenced) and 1s (value referenced)
        for value_name in llm_result:  # get value objects for each value name returned by the llm
            for value in self.values.values():
                if value.name == value_name:
                    extracted_values.append(value)

        print(f"For the message: '{message}', the following values were extracted by the LLM: {llm_result}")
        logging.info(f"For the message: '{message}', the following values were extracted by the LLM: {llm_result}")
        return extracted_values, extracted_values_as_vec

    def prompt(self, message, single_label):
        """
        Refer message to chosen prompt strategy.
        Args:
            message (str): Input message for which values should be extracted.
        Returns:
            list: List of value names of the extracted values.
            single_label: Indication, whether to predict a single value or a subset of values
        """
        if self.prompt_strategy == 1:
            return self.single_value(message)

        elif self.prompt_strategy == 2:
            return self.single_value_description(message)

        elif self.prompt_strategy == 3:
            return self.single_prompt(message)

        elif self.prompt_strategy == 6:
            return self.conversation(message)

        elif self.prompt_strategy == 7:
            if single_label:
                return self.ollama_single_label(message)
            else:
                return self.ollama_single_value(message)

        elif self.prompt_strategy == 8:
            return self.ollama_single_prompt(message)

    def single_value(self, message):

        extracted_values = []
        for value in self.values.values():
            if not self.history_on:
                self.history = []

            prompt = PROMPT_TEMPLATES["single_value"].format(message, value.name, value.name)
            self.history.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model="local-model", messages=self.history, temperature=0.7, stream=True)

            response = next(completion).choices[0].delta.content  # get first element from stream object
            if response == "Yes":
                extracted_values.append(value.name)

            # logging.info(f" Response {response}; Value {value.name}; Message {message}")

            if self.history_on:
                complete_response = response
                message_log = {"role": "assistant", "content": ""}
                for chunk in completion:  # retrieve complete llm response
                    if chunk.choices[0].delta.content:
                        complete_response += chunk.choices[0].delta.content
                        message_log["content"] += chunk.choices[0].delta.content

                self.history.append(message_log)

        return extracted_values

    def single_value_description(self, message):
        # ToDo
        return []

    def single_prompt(self, message):
        if not self.history_on:
            self.history = []

        value_names = [value.name for value in self.values.values()]
        values_as_string = ', '.join(value_names)

        prompt = PROMPT_TEMPLATES["single_prompt"].format(message, values_as_string)
        self.history.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model="local-model", messages=self.history, temperature=0.7, stream=True)

        response = ""
        end_of_value_list_reached = False
        for chunk in completion:
            if end_of_value_list_reached:
                break
            if "]" in chunk.choices[0].delta.content:
                end_of_value_list_reached = True  # do not break immediately because last value would get excluded
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        if self.history_on:
            message_log = {"role": "assistant", "content": response}
            self.history.append(message_log)

        # make list form value string
        response = response[1:-1]
        extracted_values = [item.strip() for item in response.split(',')]

        return extracted_values

    def check_prompt_count(self):
        if self.prompt_count > 50:
            self.client = OpenAI(base_url="http://localhost:5002/v1", api_key="lm-studio")

    def zero_shot_classification(self, message):

        # prompt = "Can anyone tell me where you can learn how to drive on a mechanic? No strain of old school driving instructors"

        value_names = []
        for value in self.values.values():
            value_names.append(value.name)

        value_names.append("other")
        print(value_names)

        res = self.pipeline_classifier(message, candidate_labels=value_names)
        print(res["labels"])
        print(res["scores"])

        return []

    def question_answering(self, message):
        print("Message: ", message)

        for value in self.values.values():
            context = PROMPT_TEMPLATES["context"].format(message, value.name)
            question = PROMPT_TEMPLATES["question_answering"].format(value.name)

            res = self.pipeline_qa(question=question, context=context)
            print(value.name)
            print(res)

        return []

    def conversation(self, message):
        print("Message: ", message)

        value = "shelter"
        prompt = PROMPT_TEMPLATES["context_with_question"].format(message, value, value)

        conversation = Conversation(prompt)
        conversation = self.pipeline_conv(conversation)
        res = conversation.messages[-1]["content"]

        print(res)

        return []

    def ollama_single_label(self, message):
        """
        Two prompts: --> single-label classification
        1. prompt, asking which one value is referenced
        2. prompt, validating chosen value by asking if value is referenced
        """
        logging.info("Message: ", message)

        value_names = [value.name for value in self.values.values()]
        values_as_string = ', '.join(value_names)

        # 1. prompt
        prompt1 = PROMPT_TEMPLATES["single_label"].format(message, values_as_string)
        msgs = [
            {"role": "system",
             "content": self.context},
            {"role": "user", "content": prompt1}
        ]
        output = ollama.chat(
            model=self.ollama_model_name,
            messages=msgs
        )

        response = output['message']['content']
        response = response.lower()

        # for each value, check if the value name appears in the response
        extracted_value = ""
        for value in value_names:
            if re.search(r'\b' + re.escape(value) + r'\b', response):
                extracted_value = value
                break

        if extracted_value == "":  # the model did not choose any value, despite being asked to
            logging.warning("The model did not choose any value, despite being asked to.")
            return []

        # 2. prompt
        prompt2 = PROMPT_TEMPLATES["single_value_validation"].format(message, extracted_value)

        msgs = [
            {"role": "system",
             "content": self.context},
            {"role": "user", "content": prompt2}
        ]
        output = ollama.chat(
            model=self.ollama_model_name,
            messages=msgs
        )

        response = output['message']['content']
        if response.lower() == "yes":
            logging.info("Extracted value: ", extracted_value)
            return [extracted_value]

        return []  # if no value was extracted, return empty list

    def ollama_single_value(self, message):
        """
        Individual prompt for each value --> multi-label classification
        """
        logging.info("Message: ", message)
        extracted_values = []
        for value in self.values.values():
            prompt = PROMPT_TEMPLATES["single_value"].format(message, value.name, value.name)

            msgs = [
                {"role": "system",
                 "content": self.context},
                {"role": "user", "content": prompt}
            ]
            output = ollama.chat(
                model=self.ollama_model_name,
                messages=msgs
            )

            response = output['message']['content']
            if response.lower() == "yes":
                extracted_values.append(value.name)

            logging.info(f"Value {value.name}; Response {response}")
        return extracted_values

    def ollama_single_prompt(self, message):
        """
        One prompt for all values --> multi-label classification
        """
        logging.info("Message: ", message)

        value_names = [value.name for value in self.values.values()]
        values_as_string = ', '.join(value_names)
        prompt = PROMPT_TEMPLATES["single_prompt"].format(message, values_as_string)

        msgs = [
            {"role": "system",
             "content": self.context},
            {"role": "user", "content": prompt}
        ]
        output = ollama.chat(
            model=self.ollama_model_name,
            messages=msgs
        )

        response = output['message']['content']

        extracted_values = []
        # make list form value string
        # extracted_values = [item.strip() for item in response.split(',')]

        logging.info("Extracted values: ", extracted_values)
        return extracted_values

    def ollama_two_turn(self, message):
        logging.info("Message: ", message)

        value_names = [value.name for value in self.values.values()]
        values_as_string = ', '.join(value_names)
        prompt = PROMPT_TEMPLATES["single_prompt"].format(message, values_as_string)

        msgs = [
            {"role": "system",
             "content": PROMPT_TEMPLATES["context"]},
            {"role": "user", "content": prompt}
        ]
        output = ollama.chat(
            model=self.ollama_model_name,
            messages=msgs
        )

        response = output['message']['content']
        print(prompt)
        print(response)

        extracted_values = []
        # make list form value string
        # extracted_values = [item.strip() for item in response.split(',')]

        logging.info("Extracted values: ", extracted_values)
        return extracted_values
