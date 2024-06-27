import os
import logging
import random
from gensim.models import Word2Vec, KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PersonalValueAgent.value_extraction_sources.i_value_extraction_source import IValueExtractionSource
from PersonalValueAgent.db_utils.database_utils import DAO


def init_word_to_vec_model():
    """
    Initialize and save the Word2Vec model for word embeddings. This is a static method because only one word embedding
    is necessary even if there were multiple Dictionary instances.
    """
    messages = Dictionary.db_helper.get_messages()
    tokenized_messages = [word_tokenize(message) for message in messages]

    model = Word2Vec(sentences=tokenized_messages, window=5, min_count=1, workers=4)
    word_vectors = model.wv

    word_vectors.save(Dictionary.model_path)  # save word vectors


class Dictionary(IValueExtractionSource):
    """
    This class provides a dictionary - essentially a keyword matching algorithm - to extract values from messages. The
    starting point for the dictionary are the keywords that are contained in the value representations.

    Args:
        values (dict): Dictionary of value representations with value id as key.
        top_n_similar (int): Number of top similar words to consider during dictionary expansion.

    Attributes:
        values (dict): Dictionary of value representations with value id as key.
        keyword_dict (dict): Dictionary containing keywords for each value with value id as key.
        expanded_dict (dict): Expanded dictionary containing all keywords from "keyword_dict" and keywords added during
        expansion.
        top_n_similar (int): Number of top similar words to consider during dictionary expansion.
    """

    # local path to directory with word embedding model
    model_directory = "/Users/raoulbrigola/Documents/Documents/AIMaster/Thesis/MasterThesis/" \
                      "Value_Extraction/PersonalValueAgent/models"

    # file name of the word embedding model
    model_filename = "embedding_model.bin"

    # full local path to the model file
    model_path = os.path.join(model_directory, model_filename)

    # helper that provides functions to access database
    db_helper = DAO()

    def __init__(self, values, top_n_similar=5):
        self.values = values
        self.keyword_dict = {key: value.keywords for key, value in values.items()}
        self.expanded_dict = {}
        self.top_n_similar = top_n_similar
        self.lemmatizer = WordNetLemmatizer()

        self.add_axies_exploration_keywords()

        if not os.path.exists(Dictionary.model_path):
            init_word_to_vec_model()
        self.expand_dict()  # expand initial keyword to be able to extract more values

    def expand_dict(self):
        """
        Expand the initial keyword dictionary. All available expansion methods are called and the results combined to
        update the expanded dictionary ("self.expanded_dict")
        """
        word_to_vec_dict = self.word_to_vec_similarities()
        synonyms_dict = self.get_synonyms()

        # merge dictionaries from word2vec and from synonyms expansion
        merged_dict = word_to_vec_dict
        for key, value in synonyms_dict.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
                merged_dict[key] = list(set(merged_dict[key]))

        self.expanded_dict = merged_dict

    def get_synonyms(self):
        """
        Use nltk wordnet to get synonyms for all keywords individually. Add all found synonyms together with the
        original keywords to the expanded dictionary.
        Returns:
            dict: Expanded dictionary
        """
        expanded_dict = {}

        for value_id, keywords in self.keyword_dict.items():
            expanded_keywords = list(set(keywords))  # keep original keywords in expanded dict
            for keyword in keywords:
                for syn in wordnet.synsets(keyword):
                    for lemma in syn.lemmas():
                        expanded_keywords.append(lemma.name())

            expanded_keywords = [keyword.replace("_", " ") for keyword in expanded_keywords]
            expanded_dict[value_id] = list(set(expanded_keywords))

        return expanded_dict

    def set_top_n_similar(self, top_n_similar):
        """
        Reset the number of top similar words to consider during dictionary expansion and recalculate the dictionary
        accordingly.
        Args:
            top_n_similar (int): Number of top similar words to consider
        """
        self.top_n_similar = top_n_similar
        self.expand_dict()

    def word_to_vec_similarities(self):
        """
        Expand the keyword dictionary by calculate the top_n_similar words for each keyword using the Word2Vec
        word embedding model.
        Returns:
            dict: Expanded dictionary
        """
        # Quote from gensim documentation (https://radimrehurek.com/gensim/models/word2vec.html):
        # The reason for separating the trained vectors into KeyedVectors is that if you don’t need the full model
        # state any more (don’t need to continue training), its state can be discarded, keeping just the vectors and
        # their keys proper.
        word_vectors = KeyedVectors.load(Dictionary.model_path)

        # calculate similarity and select similar words for each keyword for each value
        expanded_dict = {}
        for value_id, keywords in self.keyword_dict.items():
            expanded_keywords = set(keywords)  # keep original keywords in expanded dict
            for keyword in keywords:
                if keyword in word_vectors:
                    similar_words = word_vectors.similar_by_word(keyword, topn=self.top_n_similar)

                    for word, _ in similar_words:
                        if word not in STOPWORDS:  # filter out stop words (look at import)
                            expanded_keywords.add(word)

            expanded_dict[value_id] = list(expanded_keywords)

        return expanded_dict

    def get_values_for_message(self, message, single_label):
        """
        Get values corresponding to keywords found in the message. A value is found if at least one corresponding
        keyword appears in the message. This function essentially implements a keyword matching algorithm that uses
        lemmatization to improve the performance.
        Args:
            message (str): Input message for which values should be extracted.
        Returns:
            list: List of extracted values.
        """
        extracted_values = []
        extracted_values_as_vec = []  # vector representation with 0s (value not referenced) and 1s (value referenced)
        for value_id, keywords in self.expanded_dict.items():
            value_is_referenced = False
            for keyword in keywords:
                keyword_lemma = self.lemmatizer.lemmatize(keyword)  # lemmatization
                message_lemma = self.lemmatizer.lemmatize(message)

                if keyword_lemma in message_lemma:
                    extracted_values.append(self.values[value_id])
                    extracted_values_as_vec.append(1)
                    value_is_referenced = True
                    break  # break because only one keyword has to be found for each value

            if not value_is_referenced:
                extracted_values_as_vec.append(0)

        if single_label and len(extracted_values) > 1:
            random_value = random.choice(extracted_values)
            extracted_values = [random_value]

        logging.info(f"For the message: '{message}', the following values were extracted by the dictionary: "
                     f"{[value.name for value in extracted_values]}")
        return extracted_values, extracted_values_as_vec

    def add_axies_exploration_keywords(self):
        if len(self.keyword_dict) > 0:
            self.keyword_dict[1] = ["housing"]
        if len(self.keyword_dict) > 5:
            self.keyword_dict[6].append("service")
            self.keyword_dict[6].append("health")
        if len(self.keyword_dict) > 8:
            self.keyword_dict[9].append("package")
        if len(self.keyword_dict) > 10:
            self.keyword_dict[11].append("general practitioner")
        if len(self.keyword_dict) > 11:
            self.keyword_dict[12].append("dog")
        if len(self.keyword_dict) > 17:
            self.keyword_dict[18].append("Red Cross")


