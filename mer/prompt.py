import copy
import json
import random
from abc import ABC, abstractmethod
from mer.utils import calculate_wer


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

    @classmethod
    def from_txt(cls, txt_path, **kwargs):
        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()
        return cls(txt, **kwargs)

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
        self.txt = txt
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

    def __init__(self, txt, config=None, simple=False, seed=10):
        self.config = txt
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
        minor = int(error_count_dict["minor"])
        standard = int(error_count_dict["standard"])
        serious = int(error_count_dict["serious"])
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

    @staticmethod
    def convert_examples_to_dict(txt):
        """
        format a txt input prompt as needed for GPT input
        """

        examples = []
        for item in txt:
            example = {}
            error_types = []
            reasons = []
            for line in item.split("\n"):
                key, value = line.split(":")
                key = key.lower()
                value = value.strip()

                if key == "error":
                    error_types = value.split()

                elif key in ["reference", "recognised"]:
                    example[key] = value
                elif key == "reason":
                    reasons = value.split(".")
                    reasons = [
                        f"{reason.strip()}." if not reason.endswith(".") else reason.strip() for reason in reasons
                    ]
                else:
                    exit(f"Text file contains an unrecognised keyword: {key}")

            errors = [{"reason": reason, "error_type": error_type} for reason, error_type in zip(reasons, error_types)]
            example["errors"] = errors

            examples.append(example)

        return examples

    def get_prompt_base(self):
        """Build the base prompt which has the error descriptions followed by the few shot examples"""

        if isinstance(self.config, str):
            base, *examples = self.config.split("\n\n")
            base += "\n\n"
            examples = self.convert_examples_to_dict(examples)
        else:
            base = self.config["base"]
            base += "\n\n"
            examples = self.config["examples"]

        for example in examples:
            _, _, results = calculate_wer(example["reference"], example["recognised"])

            base += f"Comparison:{results['comparison']}\n"
            base += "Output:\n"
            base += "[\n"

            label = []

            for error in example["errors"]:
                reason = error["reason"].replace('"', "'")
                label.append(json.dumps({"reason": reason, "error_type": error["error_type"]}))

            base += ",\n".join(label)
            base += "\n]\n\n"

        return base

    def get_score_mapping(self):
        error2score = {"minor": 0.25, "standard": 0.5, "serious": 1}
        return error2score

    def create_prompt(self, ref, rec):
        _, _, wer_result = calculate_wer(ref, rec)
        comparison = wer_result["comparison"]
        return f"""{copy.deepcopy(self.base)}Comparison:{comparison}
Output:"""

    def get_result(self, text):
        assert text is not None, "Text is empty"
        try:
            output = json.loads(text)
        except json.decoder.JSONDecodeError:
            print(f"Bad JSON from LM. Can't decode '{text}'")
            return None, None

        error_count_dict = {"minor": 0, "standard": 0, "serious": 0, "reason": []}

        error_string_map = {"minor": "m", "standard": "s", "serious": "e"}
        error_str = ""
        for row in output:
            try:
                # Get reason from second line and result (containing error counts) on the fourth line
                reason, error = row["reason"], row["error_type"].strip()
                # Count the number of errors
                error_count_dict["reason"].append(reason)
                error_count_dict[error] += 1
                error_str += error_string_map[error]

            except IndexError:
                print(f"Bad continuation from LM as can't unpack items {text}")
                return None, None

        penalty_from_counts = self.get_penalty(error_count_dict)
        error_count_dict["reason"] = " ".join(error_count_dict["reason"])

        # return error_count_dict, penalty_from_counts
        return error_count_dict, error_str
