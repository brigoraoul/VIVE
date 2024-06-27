from abc import ABC, abstractmethod


class IValueExtractionSource(ABC):

    @abstractmethod
    def get_values_for_message(self, message, single_label) -> list:
        pass



