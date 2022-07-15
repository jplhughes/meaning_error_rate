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
