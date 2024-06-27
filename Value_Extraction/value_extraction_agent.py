import logging

from PersonalValueAgent.value_extraction_sources.llm import LLM
from PersonalValueAgent.value_extraction_sources.dictionary import Dictionary
from PersonalValueAgent.value_extraction_sources.e_value_extraction_source import EValueExtractionSource
from PersonalValueAgent.db_utils.database_utils import DAO
from PersonalValueAgent.db_utils.models.value import Value

logging.basicConfig(level=logging.WARNING)


class ValueExtractionAgent:
    """
    Instances of this class function as agents that can find references to personal values in natural language text
    messages. Given new messages, an agent returns personal values from a pre-defined list that it thinks have
    motivated the writer to compose the message. To create the list of personal values for a message the agent uses
    different "value extraction sources".

    Attributes:
        db_helper (DAO): Helper class to get value representations from the database.
        values (dict): Dictionary of value representations obtained from the database.
        value_extraction_sources (dict): Dictionary of sources that the agent can use to predict values for messages.
    """

    def __init__(self):
        self.db_helper = DAO()
        self.values = self.db_helper.get_value_representations()

        # determines the
        self.optimize_for_recall = True

        # default sources can be overwritten by "set_value_extraction_sources"
        self.value_extraction_sources = {
            "LLM": LLM(self.values, 1),
            "Dictionary": Dictionary(self.values)
        }

    def get_values_for_message(self, message, single_label=False):
        """Extract values from a single message by using all available value extraction sources. When optimizing for
        recall, take the union of the predicted values of all value extraction sources. When optimizing for precision,
        take intersection of the predicted values of all value extraction sources.
        Args:
            message (str): The message from which values are to be extracted.
        Returns:
            list: List of extracted values.
            single_label: Indication, whether to predict a single value or a subset of values
        """
        combined_predicted_values = set()
        extracted_single_values_as_vec = []
        extracted_single_values = []
        for source in self.value_extraction_sources.values():
            extracted_values, extracted_values_as_vec = source.get_values_for_message(message, single_label)
            extracted_single_values.append(extracted_values)
            extracted_single_values_as_vec.append(extracted_values_as_vec)

            # multi-label case
            if self.optimize_for_recall:
                # when optimizing for recall, keep all values extracted by any source
                combined_predicted_values = combined_predicted_values.union(extracted_values)
            else:
                # when optimizing for precision, keep only values extracted by each source
                combined_predicted_values = combined_predicted_values.intersection(extracted_values)

        if single_label and len(extracted_single_values) == 2:  # single-label case for two value extraction sources
            if extracted_single_values_as_vec[0] == extracted_single_values_as_vec[1]:
                combined_predicted_values = extracted_single_values[0]
            elif sum(extracted_single_values_as_vec[0]) == 1 and sum(extracted_single_values_as_vec[1]) == 0:
                combined_predicted_values = extracted_single_values[0]
            elif sum(extracted_single_values_as_vec[0]) == 0 and sum(extracted_single_values_as_vec[1]) == 1:
                combined_predicted_values = extracted_single_values[1]
            elif sum(extracted_single_values_as_vec[0]) == 0 and sum(extracted_single_values_as_vec[1]) == 0:
                combined_predicted_values = []

        return list(combined_predicted_values)

    def get_values_for_messages(self, messages, single_label=False):
        """Extract values from multiple messages.
        Args:
            messages (list): List of messages from which values are to be extracted.
            single_label: Indication, whether to predict a single value or a subset of values
        Returns:
            dict: Dictionary with messages as keys and lists of extracted values as values
        """
        predicted_values = {}
        for message in messages:
            predicted_values[message] = self.get_values_for_message(message, single_label=single_label)

        return predicted_values

    def set_value_extraction_sources(self, value_extraction_source_names):
        """(Re)set value extraction sources of the agent. All previously existing sources have to be reinitialized.
        Value extraction sources are created through the value extraction source enum at
        PersonalValueAgent.value_extraction_sources.e_value_extraction_source.py.
        Args:
            value_extraction_source_names (list): List of value extraction source names.
        """
        value_extraction_sources = {}

        for source_name in value_extraction_source_names:
            if source_name in EValueExtractionSource.__members__:
                dict_key = source_name
                if "LLM" in source_name:
                    dict_key = "LLM"
                value_extraction_sources[dict_key] = EValueExtractionSource[source_name].value[1](self.values)
            else:
                logging.warning(f"Not a valid value extraction source: '{source_name}'")

        self.value_extraction_sources = value_extraction_sources

    def set_values(self, values):
        """(Re)set the personal values of the agent that provide info as to what the agent should look for in new
        messages.
        Args:
            values (dict): Dictionary with value IDs as keys and value objects as values.
        """
        all_value_objects = True
        for value in values.values():
            if not isinstance(value, Value):
                all_value_objects = False
                logging.warning("When setting the personal values of the agent, pass a dictionary of value objects.")
                break

        if all_value_objects:
            self.values = values

    def get_llm(self):
        """Get the LLM value extraction source instance if it exists.
        Returns:
            LLM: The LLM value extraction source instance.
        """
        for source_name in self.value_extraction_sources.keys():
            if "LLM" in source_name:
                return self.value_extraction_sources[source_name]

        logging.warning("No LLM in value extraction sources.")

    def get_dictionary(self):
        """Get the Dictionary value extraction source instance if it exists.
        Returns:
            Dictionary: The Dictionary value extraction source instance.
        """
        if "Dictionary" in self.value_extraction_sources:
            return self.value_extraction_sources["Dictionary"]
        else:
            logging.warning("No dictionary in value extraction sources.")
