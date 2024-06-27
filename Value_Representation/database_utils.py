import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models.value import Value


DATABASE_URL = "sqlite:///datasource/axies.db"

database_path = os.path.abspath("datasource/axies.db")
DATABASE_URL = f"sqlite:///{database_path}"
db = create_engine(DATABASE_URL)


def add_exploration_values(values):
    """
    Adding keywords that were annotated during exploration to the consolidation values / keywords.
    Args:
        values: dict of consolidation values + keywords
    Returns: Dictionary of class "Value" objects
    """
    query = text('SELECT * FROM keyword')
    keywords = session.execute(query).fetchall()

    for keyword_tuple in keywords:
        values[keyword_tuple[3]].add_keyword(keyword_tuple[1])

    return values


def get_consolidation_values():
    """
    Getting all values that were approved by the annotator during consolidation + getting respective keywords from
    consolidation.
    Returns: Dictionary of class "Value" objects
    """
    query = text('SELECT * FROM consolidation_value')
    value_tuples = session.execute(query).fetchall()
    query = text('SELECT * FROM consolidation_keyword')
    keywords = session.execute(query).fetchall()

    values = {}
    for value_tuple in value_tuples:
        value_id = value_tuple[0]
        value_name = value_tuple[1]
        value_description = value_tuple[5]

        if not value_description:
            values[value_id] = Value(value_id, value_name)
        else:
            for value in values.values():
                if value.name == value_name:
                    value.change_description(value_description)

    for keyword_tuple in keywords:
        values[keyword_tuple[4]].add_keyword(keyword_tuple[1])

    return values


session = Session(db)

value_dict = get_consolidation_values()
value_dict = add_exploration_values(value_dict)

for value in value_dict.values():
    value.print()

session.close()