import copy
import json


class Prompt:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.base = self.get_prompt_base()
        self.error2score = self.get_score_mapping()

    def get_prompt_base(self):
        """Build the base prompt which has the error descriptions followed by the few shot examples"""
        base = []
        for error_type in self.config["errors"]:
            description = self.config["errors"][error_type]["description"]
            base.append(f"{error_type.capitalize()} error - {description}.\n")

        for example in self.config["examples"]:
            error = example["error"]
            ref = example["reference"]
            rec = example["recognised"]
            reason = example["reason"]

            base.append(f"Reference: {ref}")
            base.append(f"Recognised: {rec}")
            base.append(f"Result: {error} due to {reason}. I hope it is correct.\n")

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
        prompt.append(f"Result:")
        return "\n".join(prompt)

    def get_result(self, text):
        assert text is not None, "Text is empty"

        # Error type should be first word and should be contained in error2score
        error_type = text.split()[0]
        if error_type in self.error2score:
            score = self.error2score[error_type]
        else:
            raise f"Got unexpected error type {error_type}"

        # Get reason by disecting the known parts expected in the continuation
        try:
            reason = text.split["due to "][1].split[". I hope it is correct"][0]
        except:
            reason = (
                text.replace(error_type, "")
                .replace("due to", "")
                .replace("I hope it is correct", "")
                .replace(".", "")
                .strip()
            )

        return error_type, reason, score
