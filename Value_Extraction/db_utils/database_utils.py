import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from PersonalValueAgent.db_utils.models.value import Value


class DAO:

    def __init__(self):

        self.database_path = "/Users/raoulbrigola/Documents/Documents/AIMaster/Thesis/MasterThesis/Value_Representation/datasource/axies.db"
        self.DATABASE_URL = f"sqlite:///{self.database_path}"
        self.db = create_engine(self.DATABASE_URL)  # echo=True to turn on logging

        self.session = None

    def get_exploration_values(self, values):
        """
        Adding keywords that were annotated during exploration to the consolidation values / keywords.
        Args:
            values: dict of consolidation values + keywords
        Returns: Dictionary of class "Value" objects
        """
        query = text('SELECT * FROM keyword')
        keywords = self.session.execute(query).fetchall()

        for keyword_tuple in keywords:
            values[keyword_tuple[3]].add_keyword(keyword_tuple[1])

        return values

    def get_consolidation_values(self):
        """
        Getting all values that were approved by the annotator during consolidation + getting respective keywords from
        consolidation.
        Returns: Dictionary of class "Value" objects
        """
        query = text('SELECT * FROM consolidation_value')
        value_tuples = self.session.execute(query).fetchall()
        query = text('SELECT * FROM consolidation_keyword')
        keywords = self.session.execute(query).fetchall()

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

    def get_value_representations(self):
        if os.path.exists(self.database_path):
            if self.session is None:
                self.session = Session(self.db)

            value_dict = self.get_consolidation_values()
            value_dict = self.get_exploration_values(value_dict)

            self.session.close()
            self.session = None
        else:
            print(f"Error: Database file not found at {self.database_path}")

        return value_dict


    def get_messages(self):
        if os.path.exists(self.database_path):
            if self.session is None:
                self.session = Session(self.db)

                # query all messages that were eligible for value identification and start filtering from there
                query = text("SELECT motivation_en FROM motivation")
                messages = self.session.execute(query).fetchall()
                messages = [message[0] for message in messages]

            self.session.close()
            self.session = None
        else:
            print(f"Error: Database file not found at {self.database_path}")

        return messages

