import random
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def create_labeling_sheet(doc_name):
    """
    Create an excel sheet that can be used to label messages. Sample messages randomly as to not keep original order
    and categorization.
    Args:
        doc_name (String): Name of the excel file to which the sampled messages are written for labeling.
    """
    # querying db from value identification for data for evaluation
    database_path = ""  # insert local database path
    DATABASE_URL = f"sqlite:///{database_path}"
    db = create_engine(DATABASE_URL)  # echo=True to turn on logging

    session = Session(db)

    # query all messages that were eligible for value identification and start filtering from there
    query = text("SELECT motivation_en FROM motivation")
    messages = session.execute(query).fetchall()
    messages = [message[0] for message in messages]

    session.close()

    # sample messages
    sample_size = len(messages)-1
    sample = random.sample(messages, sample_size)
    df = pd.DataFrame(sample, columns=["Message"])

    # write excel file for labeling
    labels = pd.DataFrame(columns=["shelter", "mental health", "staying connected",
                                   "disappointment in this city/country", "help for refugees"])

    df.to_excel(doc_name)


def read_labeling_sheet(doc_name):
    """
    Reading data from a filled out labeling excel sheet.
    Args:
        doc_name: File directory of the labeling sheet.
    Returns:
        list: Test data where each entry in the list is a dictionary containing the annotated value names and the
        corresponding message.
        list: List of messages that were annotated.
    """
    df = pd.read_excel(doc_name)
    df = df.iloc[:181, 1:8]

    df = df[df.iloc[:, 0] != "Message"]  # remove header rows
    df = df.rename(columns={'Unnamed: 7': 'no label'})  # rename last column

    test_data = []
    for index, row in df.iterrows():
        values = row.index[row.notna()].tolist()  # get all column names for which entry in row is not nan
        if len(values) == 1:
            test_data.append({"Values": ["no label"], "Message": row.tolist()[0]})
        else:
            test_data.append({"Values": values[1:], "Message": row.tolist()[0]})

    return df, test_data, df["Message"].tolist()


def summarize_labeling(doc_name):
    df = pd.read_excel(doc_name)
    df = df.iloc[:181, 1:8]

    df = df[df.iloc[:, 0] != "Message"]  # remove header rows
    df = df.rename(columns={'Unnamed: 7': 'no label'})  # rename last column

    labels_per_class = df.eq('x').sum()  # number of labels for each values (e.g. 27 messages annotated with "shelter")
    print(labels_per_class)

    total_annotations = labels_per_class.sum()  # number of values, NOT messages
    print("Total number of values labeled (including 'no label'): ", total_annotations)

    count_multiple = ((df == "x").sum(axis=1) > 1).sum()  # number of messages with multiple annotations
    print("Number of messages with multiple values:", count_multiple)


def confusion_matrx(annotation_1, annotation_2):
    """
    Calculate confusion matrix between two annotations.
    """
    labels_1 = [m["Values"][0] for m in annotation_1]
    labels_2 = [m["Values"][0] for m in annotation_2]

    cm = confusion_matrix(labels_1, labels_2, labels=["shelter", "mental health", "staying connected",
                                   "disappointment in this city/country", "help for refugees", "no label"])
    return cm


def cohen_kappa(annotation_1, annotation_2):
    """
    Using sklearn to calculate the cohen kappa for two value annotations.
    Args:
        annotation_1: Annotation from annotator 1
        annotation_2: Annotation from annotator 2
    Returns:
         float: cohen kappa score in range [0;1]
    """
    labels_1 = [m["Values"][0] for m in annotation_1]
    labels_2 = [m["Values"][0] for m in annotation_2]

    score = cohen_kappa_score(labels_1, labels_2)
    return score


def intercoder_agreement():
    """
    Evaluation of message labelling with Cohen's Kappa and confusion matrix of annotations.
    """
    df_ek, test_data_ek, test_messages_ek = read_labeling_sheet("")  # insert names of locally stored annotation sheets
    df_pa, test_data_pa, test_messages_pa = read_labeling_sheet("")

    cm = confusion_matrx(test_data_ek, test_data_pa)
    print(cm)
    cohen_kappa_ = cohen_kappa(test_data_ek, test_data_pa)
    print(cohen_kappa_)

