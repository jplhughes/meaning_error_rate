import copy
import json
import random
from abc import ABC, abstractmethod


class PromptBase(ABC):
    """
    Prompt object that generates the LM prompt given a config file.
    It can also find the result given the LM output.
    """

    @classmethod
    def from_file(cls, config_path, **kwargs):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config, **kwargs)

    @staticmethod
    @abstractmethod
    def unpack_example(example):
        pass

    @abstractmethod
    def get_prompt_base(self):
        pass

    @abstractmethod
    def get_score_mapping(self):
        pass

    @abstractmethod
    def create_prompt(self, ref, rec):
        pass

    @abstractmethod
    def get_result(self, text):
        pass


class Prompt(PromptBase):
    """
    Prompt object that generates the LM prompt given a config file.
    It can also find the result given the LM output.
    """

    def __init__(self, config, simple=False, seed=10):
        self.config = config
        self.simple = simple
        random.seed(seed)
        # Create prompt based on config
        self.base = self.get_prompt_base()
        self.error2score = self.get_score_mapping()

    @staticmethod
    def unpack_example(example):
        error = example.get("error", None)
        reason = example.get("reason", None)
        return error, example["reference"], example["recognised"], reason

    def get_prompt_base(self):
        """Build the base prompt which has the error descriptions followed by the few shot examples"""
        base = []
        if self.simple:
            # Just enumerate errors in prompt
            errors = self.config["errors"].keys()
            errors_joined = ", ".join(errors)
            base.append(f"Classify the severity of error out of {len(errors)} categories: {errors_joined}.\n")
        else:
            # Add description of each error type into top of prompt
            for error_type in self.config["errors"]:
                description = self.config["errors"][error_type]["description"]
                base.append(f"{error_type.capitalize()} error - {description}.\n")

        random.shuffle(self.config["examples"])  # shuffle so no order to examples
        for example in self.config["examples"]:
            error, ref, rec, reason = self.unpack_example(example)

            base.append(f"Reference: {ref}")
            base.append(f"Recognised: {rec}")
            base.append(f"Result: {reason}. Therefore, the error is likely {error}.\n")

        return base

    def get_score_mapping(self):
        error2score = {}
        for error_type in self.config["errors"]:
            error2score[error_type] = self.config["errors"][error_type]["score"]
        return error2score

    def create_prompt(self, ref, rec):
        prompt = copy.deepcopy(self.base)
        prompt.append(f"Reference: {ref}")
        prompt.append(f"Recognised: {rec}")
        prompt.append("Result:")
        return "\n".join(prompt)

    def get_result(self, text):
        assert text is not None, "Text is empty"

        # Error type should be first word and should be contained in error2score
        error_type = text.strip().split()[-1].replace(".", "")
        if error_type in self.error2score:
            score = self.error2score[error_type]
        else:
            # Don't penalise an unknown error (unlikely to get through majority voting anyway)
            error_type = "unknown"
            score = 0.0
            print(f"WARNING: Got unexpected error type {error_type} in text {text}")

        # Get reason by disecting the known parts expected in the continuation
        reason = text.split(". Therefore")[0].strip()

        return error_type, reason, score


class PromptMultiple(PromptBase):
    """
    Prompt object that generates the LM prompt given a config file.
    It can also find the result given the LM output.
    """

    def __init__(self, config, simple=False, seed=10):
        self.config = config
        self.simple = simple
        random.seed(seed)
        # Create prompt based on config
        self.error2score = self.get_score_mapping()
        self.base = self.get_prompt_base()

    @staticmethod
    def unpack_example(example):
        if example.get("minor", None) is not None:
            error_count_dict = {
                "minor": example.get("minor"),
                "standard": example.get("standard"),
                "serious": example.get("serious"),
                "reason": example.get("reason", None),
            }
        else:
            error_count_dict = None
        return error_count_dict, example["reference"], example["recognised"]

    @staticmethod
    def unpack_error_counts(error_count_dict):
        minor = error_count_dict["minor"]
        standard = error_count_dict["standard"]
        serious = error_count_dict["serious"]
        reason = error_count_dict["reason"]
        return minor, standard, serious, reason

    def get_penalty(self, error_count_dict):
        minor, standard, serious, _ = self.unpack_error_counts(error_count_dict)
        penalty = (
            minor * self.error2score["minor"]
            + standard * self.error2score["standard"]
            + serious * self.error2score["serious"]
        )
        return penalty

    def get_prompt_base(self):
        """Build the base prompt which has the error descriptions followed by the few shot examples"""
        base = []
        if self.simple:
            # Just enumerate errors in prompt
            errors = self.config["errors"].keys()
            errors_joined = ", ".join(errors)
            base.append(f"Classify the severity of error out of {len(errors)} categories: {errors_joined}.\n")
        else:
            # Add description of each error type into top of prompt
            for error_type in self.config["errors"]:
                description = self.config["errors"][error_type]["description"]
                base.append(f"{error_type.capitalize()} error - {description}.\n")
            base.append(
                "Disfluences, hyphens joining words, equivalent numbering and correct contractions can be ignored completely.\n"
            )

        random.shuffle(self.config["examples"])  # shuffle so no order to examples
        for example in self.config["examples"]:
            error_count_dict, ref, rec = self.unpack_example(example)
            penalty = self.get_penalty(error_count_dict)
            minor, standard, serious, reason = self.unpack_error_counts(error_count_dict)

            base.append(f"Reference: {ref}")
            base.append(f"Recognised: {rec}")
            base.append(f"Reasoning: {reason}")
            base.append(f"Result: {minor} minor + {standard} standard + {serious} serious = {penalty} penalty\n")

        return base

    def get_score_mapping(self):
        error2score = {}
        for error_type in self.config["errors"]:
            error2score[error_type] = self.config["errors"][error_type]["score"]
        return error2score

    def create_prompt(self, ref, rec):
        prompt = copy.deepcopy(self.base)
        prompt.append(f"Reference: {ref}")
        prompt.append(f"Recognised: {rec}")
        prompt.append("Reasoning:")
        return "\n".join(prompt)

    def get_result(self, text):
        assert text is not None, "Text is empty"
        lines = text.strip().split("\n")
        try:
            # Get reason from first line and result (containing error counts) on the second line
            reason, result = lines[0], lines[1]
            # Unpack the counts from result line
            # e.g. Result: 1 minor + 0 standard + 1 serious = 1.25 penalty
            error_count_dict = {
                "minor": result.strip().split()[1],
                "standard": result.strip().split()[4],
                "serious": result.strip().split()[7],
                "reason": reason,
            }
            penalty_from_prompt = result.strip().split()[10]
        except IndexError:
            print(f"Bad continuation from LM as can't unpack items {text}")
            return None, None, None

        penalty_from_counts = self.get_penalty(error_count_dict)

        if penalty_from_prompt != penalty_from_counts:
            print(f"WARNING: LM bad at maths! It said {penalty_from_prompt} but should be {penalty_from_counts}.")

        return error_count_dict, penalty_from_counts
