import argparse
import copy
import json

from mer.lm import LanguageModel
from mer.prompt import Prompt


def get_accuracy(test_json, prompt_config_path, output_json):

    with open(test_json, "r") as f:
        testset = json.load(f)

    prompt = Prompt.from_file(prompt_config_path, simple=True)
    lm = LanguageModel()

    output = []
    for i, example in enumerate(testset["examples"]):
        error_type, ref, rec, reason = prompt.unpack_example(example)

        prompt_string = prompt.create_prompt(ref, rec)
        lm.print_cost(prompt_string)
        text, _ = lm.get_continuation(prompt_string)
        error_type_pred, reason_pred, _ = prompt.get_result(text)

        outcome = "incorrect"
        if error_type_pred == error_type:
            print(f"Got example {i} correct")
            outcome = "correct"

        data = {"reference": ref, "recognised": rec}
        data["target"] = {"error": error_type, "reason": reason}
        data["predicted"] = {"error": error_type_pred, "reason": reason_pred}
        data["outcome"] = outcome
        output.append(data)

    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_json",
        type=str,
        default="./config/test.json",
        help="Json file containing examples with labels",
    )
    parser.add_argument(
        "--prompt_config_path",
        type=str,
        default="./config/prompt.json",
        help="path to prompt config json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./data/results.json",
        help="path to output json to store results",
    )
    args = parser.parse_args()

    get_accuracy(args.test_json, args.prompt_config_path, args.output_json)


if __name__ == "__main__":
    main()