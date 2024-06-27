import sys
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Value_Extraction_Agent.value_extraction_agent import ValueExtractionAgent
from database_utils import DAO
from data_annotation.message_labelling import read_labeling_sheet, confusion_matrx, cohen_kappa, summarize_labeling


db_helper = DAO()

# DATABASE QUERYING

# querying db from value identification for data for evaluation
database_path = ""  # insert local database path
DATABASE_URL = f"sqlite:///{database_path}"
db = create_engine(DATABASE_URL)  # echo=True to turn on logging

session = Session(db)

# 1. getting shown motivation ids for all annotation actions where a new value was added by the annotator
query = text("SELECT shown_motivation, value FROM annotation_action WHERE action = 'ADD_VALUE'")
add_value_actions = session.execute(query).fetchall()

multi_test_data = []  # contains annotated messages + annotated values
multi_test_messages = []  # contains only annotated messages

# 2. get motivations by id plus the value that was added for it
for action in add_value_actions:
    query = text("SELECT motivation_en FROM motivation WHERE id = :id")
    message = session.execute(query, {"id": action[0]}).fetchone()

    query = text("SELECT name FROM value WHERE id = :id")
    value = session.execute(query, {"id": action[1]}).fetchone()

    if message:
        multi_test_data.append({"Value": value[0], "Message": message[0]})
        multi_test_messages.append(message[0])

session.close()


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# HELPER FUNCTIONS

def precision(true, pred):
    """
    Of all the values extracted by the agent, how many were actually annotated by the annotators?
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
    Returns:
        Precision
    """
    correctly_predicted = 0
    total_predictions = 0
    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        for v in pred_values:
            total_predictions += 1
            if v in true_values:
                correctly_predicted += 1

    if total_predictions == 0:
        return 0

    return correctly_predicted / total_predictions


def precision_per_value(true, pred, value_names):
    """
    Calculate the individual precision for each value.
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
    Returns:
        Precision
    """
    value_names.append("no label")
    correctly_predicted = {value_name: 0 for value_name in value_names}
    total_predictions = {value_name: 0 for value_name in value_names}

    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        for v in pred_values:
            total_predictions[v] += 1
            if v in true_values:
                correctly_predicted[v] += 1

        if not pred_values:
            total_predictions["no label"] += 1
            if not correctly_predicted:
                correctly_predicted["no label"] += 1

    precision_dict = {}
    for value in correctly_predicted:
        if total_predictions[value] == 0:
            precision_dict[value] = 0
        else:
            precision_dict[value] = correctly_predicted[value] / total_predictions[value]

    return precision_dict


def recall(true, pred):
    """
    Of all the annotated values in the dataset, how many were correctly extracted by the model?
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
    Returns:
        Recall
    """
    correctly_predicted = 0
    total_annotations = 0
    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        total_annotations += len(true_values)
        for v in pred_values:
            if v in true_values:
                correctly_predicted += 1

    if total_annotations == 0:
        return 0

    return correctly_predicted / total_annotations


def recall_per_value(true, pred, value_names):
    """
    Calculate the individual recall for each value.
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
    Returns:
        Precision
    """
    correctly_predicted = {value_name: 0 for value_name in value_names}
    total_annotations = {value_name: 0 for value_name in value_names}

    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        for v in true_values:
            total_annotations[v] += 1

        for v in pred_values:
            if v in true_values:
                correctly_predicted[v] += 1

    recall_dict = {}
    for value in correctly_predicted:
        if correctly_predicted[value] == 0 or total_annotations[value] == 0:
            recall_dict[value] = 0
        else:
            recall_dict[value] = correctly_predicted[value] / total_annotations[value]

    return recall_dict


def f1(prec, rec):
    if (prec + rec) == 0:
        return -1
    return 2 * (prec * rec) / (prec + rec)


def f1_per_value(precision, recall, value_names):
    """
    Calculate the individual f1 score for each value.
    Args:
        precision: precision per value
        recall: recall per value
    Returns:
        List containing F1 scores per value
    """
    f1_ = []
    for value_name in value_names:
        f1_.append(f1(precision[value_name], recall[value_name]))
    return f1_


def exact_match_acc(true, pred):
    """
    Calculates the ratio of messages for which the exact correct subset of context-specific values was extracted.
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
    Returns:
        exact match ratio: float
    """
    exact_matches = 0
    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        if sorted(pred_values) == sorted(true_values):
            exact_matches += 1

    return exact_matches / len(multi_test_data)


def accuracy(true, pred, value_names):
    """
    Calculate accuracy that takes into account partial correctness of a value prediction. For every message, calculate
    the ratio of correctly predicted and incorrectly predicted values from the set of context-specific values. This
    function only makes sense of the multi-label task.
    Args:
        true: messages annotated with values
        pred: messages with values predicted by agent
        value_names: list of names of values in the set of context-specific values
    Returns:
        partial correctness accuracy: float
    """
    correctly_predicted_values = 0
    for i in range(len(true)):
        true_values = true[i]
        pred_values = pred[i]

        message_score = 0.0
        for value_name in value_names:
            if value_name in pred_values and value_name in true_values:  # true positive prediction
                message_score = message_score + 1 / len(value_names)
            elif value_name not in pred_values and value_name not in true_values:  # true negative prediction
                message_score = message_score + 1 / len(value_names)

        correctly_predicted_values += message_score

    return correctly_predicted_values / len(multi_test_data)


def confusion_matrx(test_data, predictions, classifier_name, iteration):
    """
    1. Calculate confusion matrix with sklearn.
    2. Visualize confusion matrix.
    Args:
        test_data: messages annotated with values
        predictions: messages with values predicted by agent
    Returns:
        cm: Confusion matrix as two-dimensional list.
    """
    true = []
    pred = []
    for value_message_pair in test_data:
        message = value_message_pair["Message"]

        true_values = value_message_pair["Values"]
        true_value = true_values[0]
        true.append(true_value)

        predicted_values = predictions[message]
        if not predicted_values:
            predicted_values = ["no label"]
        else:
            predicted_values = [value.name for value in predicted_values]
        pred.append(predicted_values[0])

        cm = confusion_matrix(true, pred, labels=["shelter", "mental health", "staying connected",
                                                  "disappointment in this city/country", "help for refugees",
                                                  "no label"])

    true_arr = np.array(true)
    pred_arr = np.array(pred)

    disp = ConfusionMatrixDisplay.from_predictions(true_arr, pred_arr,
                                                   labels=["shelter", "mental health", "staying connected",
                                                   "disappointment in this city/country", "help for refugees",
                                                   "no label"], display_labels=["1", "2", "3", "4", "5", "no label"])
    disp.plot(cmap=plt.cm.Blues)
    cm_file_name = "confusion_matrices/" + classifier_name + str(iteration)
    plt.savefig(cm_file_name)

    return cm


def reformat_agent_output(test_data, predictions):
    """
    Change the format of the annotations and predictions to a more convenient format for evaluation.
    Args:
        test_data: messages annotated with values
        predictions: messages with values predicted by agent
    Returns:
         true: reformated annotations
         pred: reformated predictions
    """
    true = []
    pred = []
    for value_message_pair in test_data:
        true_values = value_message_pair["Values"]
        if "no label" in true_values:
            true_values = []
        true.append(true_values)

        message = value_message_pair["Message"]
        predicted_values = [value.name for value in predictions[message]]
        pred.append(predicted_values)

    return true, pred


def evaluate(test_data, predictions, value_names, classifier_name, iteration, single_label=True):
    """
    Calculate accuracy that takes into account partial correctness of a value prediction. For every message, calculate
    the ratio of correctly predicted and incorrectly predicted values from the set of context-specific values.
    Args:
        test_data: messages annotated with values
        predictions: messages with values predicted by agent
        value_names: list of names of values in the set of context-specific values
        single_label: boolean, indicating whether predictions are for a single-lable or multi-label task
    """
    true, pred = reformat_agent_output(test_data, predictions)

    # calculate and print evaluation metrics
    print("     Exact Match Accuracy: ", exact_match_acc(true, pred))
    if not single_label:
        print("     Partial Correctness Accuracy: ", accuracy(true, pred, value_names))
    precision_ = precision(true, pred)
    print("     Precision: ", precision_)
    recall_ = recall(true, pred)
    print("     Recall: ", recall_)
    print("     F1: ", f1(precision_, recall_))

    print("     Precision per value:")
    precision_per_value_ = precision_per_value(true, pred, value_names)
    print(precision_per_value_)
    print("     Recall per value:")
    recall_per_value_ = recall_per_value(true, pred, value_names)
    print(recall_per_value_)
    print("     F1 per value: ")
    print(f1_per_value(precision_per_value_, recall_per_value_, value_names))

    if single_label:
        print("     Confusion matrix:")
        print(confusion_matrx(test_data, predictions, classifier_name, iteration))


def intercoder_agreement():
    """
    Evaluation of message labelling with Cohen's Kappa and confusion matrix of annotations.
    """
    df_ek, test_data_ek, test_messages_ek = read_labeling_sheet("data_annotation/labeling_sheet_ekatherina.xlsx")
    df_pa, test_data_pa, test_messages_pa = read_labeling_sheet("data_annotation/labeling_sheet_paula.xlsx")

    cm = confusion_matrx(test_data_ek, test_data_pa)
    print(cm)
    cohen_kappa_ = cohen_kappa(test_data_ek, test_data_pa)
    print(cohen_kappa_)


def print_predictions(predictions):
    for message, predicted_values in predictions.items():
        print("MESSAGE: ", message)
        print("VALUES: ")
        for value in predicted_values:
            print(value.name)
        print("--------------------------------------------------------------------------------------------------")


def post_processing_dict(true, pred):
    true_filtered = {}
    pred_filtered = {}
    for message in pred:
        print(i)
        if len(pred[message]) <= 1:
            true_filtered[message] = true[message]
            pred_filtered[message] = pred[message]
        else:
            print(pred[message])

    print(len(true_filtered))
    return true_filtered, pred_filtered


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# EVALUATION OF VALUE EXTRACTION SOURCES

# reset agent values to values for which labelled data exists
values = db_helper.get_value_representations()
value_names = ["shelter", "mental health", "staying connected",
               "disappointment in this city/country", "help for refugees"]
selected_values = {key: value for key, value in values.items() if value.name in value_names}

# get test data (by default for multi label task, meaning some messages contain multiple value references)
df, multi_test_data, multi_test_messages = read_labeling_sheet("data_annotation/labeling_sheet_resoluted.xlsx")

# get test data for single label task, do not keep messages that reference more than one value
single_test_data = []
single_test_messages = []
for i in range(len(multi_test_data)):
    if len(multi_test_data[i]["Values"]) <= 1:
        single_test_data.append(multi_test_data[i])
        single_test_messages.append(multi_test_messages[i])

# evaluation agent
pv_agent = ValueExtractionAgent()
pv_agent.set_values(selected_values)
pv_agent.optimize_for_recall = True
single_label = False
iterations = 1
model_name = "llama3"
pv_agent.ollama_model_name = model_name

test_data = multi_test_data
test_messages = multi_test_messages
if single_label:
    test_data = single_test_data
    test_messages = single_test_messages

# combinations of value extraction sources for which evaluation should be done
value_extraction_source_combinations = [
    ["Dictionary"],
    ["LLM_7"],
    ["Dictionary", "LLM_7"],
]

if len(value_extraction_source_combinations) > 0:
    with open('result_log.txt', 'a') as file:
        sys.stdout = file  # write print statements in file instead of console
        print("")
        print("--------------------------------------------------------------------------------------------------")
        print(datetime.datetime.now())  # log date and time of evaluation
        print("Length of test set: ", len(test_data))

        for sources_list in value_extraction_source_combinations:
            sys.stdout = sys.__stdout__
            pv_agent.set_value_extraction_sources(sources_list)
            sys.stdout = file
            print("Model: ", model_name)
            print("Single label: ", single_label)

            for i in range(iterations):
                sys.stdout = sys.__stdout__
                print("Iteration ", i)

                start_time = time.time()
                predictions = pv_agent.get_values_for_messages(test_messages, single_label=single_label)
                end_time = time.time()
                runtime = end_time - start_time

                sys.stdout = file
                print("Runtime: ", runtime)
                print(f"Evaluation metrics for {sources_list}, Iteration {i}:")

                classifier_name = ", ".join(sources_list)
                evaluate(test_data, predictions, value_names, classifier_name, i, single_label=single_label)
