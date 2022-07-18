import copy
import json
import random


class Prompt:
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

    @classmethod
    def from_file(cls, config_path, **kwargs):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config, **kwargs)

    @staticmethod
    def unpack_example(example):
        return example["error"], example["reference"], example["recognised"], example["reason"]

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
