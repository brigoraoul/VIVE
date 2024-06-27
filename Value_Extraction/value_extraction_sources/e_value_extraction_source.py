from enum import Enum

from PersonalValueAgent.value_extraction_sources.llm import LLM
from PersonalValueAgent.value_extraction_sources.dictionary import Dictionary


class EValueExtractionSource(Enum):
    LLM_1 = ("LLM_1", lambda values: LLM(values, 1))
    LLM_2 = ("LLM_2", lambda values: LLM(values, 2))
    LLM_3 = ("LLM_3", lambda values: LLM(values, 3))
    LLM_4 = ("LLM_4", lambda values: LLM(values, 4))
    LLM_6 = ("LLM_6", lambda values: LLM(values, 6))
    LLM_7 = ("LLM_7", lambda values: LLM(values, 7))
    LLM_8 = ("LLM_8", lambda values: LLM(values, 8))
    Dictionary = ("Dictionary", lambda values: Dictionary(values))
