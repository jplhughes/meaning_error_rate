import argparse
import json
from collections import Counter

from mer.lm import LanguageModel
from mer.prompt import Prompt


def get_accuracy(test_json, prompt_config_path, output_json, api_key=None, num_samples=3):

    with open(test_json, "r", encoding="utf-8") as f:
        testset = json.load(f)

    prompt = Prompt.from_file(prompt_config_path, simple=True)
    lm = LanguageModel(api_key=api_key)

    output = []
    for i, example in enumerate(testset["examples"]):
        error_type, ref, rec, reason = prompt.unpack_example(example)
        # Add example information to item in json
        data = {"reference": ref, "recognised": rec}
        data["target"] = {"error": error_type, "reason": reason}

        # Create prompt and get continuations from lm
        prompt_string = prompt.create_prompt(ref, rec)
        lm.print_cost(prompt_string, num_samples=num_samples)  # estimated cost
        continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
        lm.print_cost(prompt_string, tokens=response["usage"]["total_tokens"], num_samples=num_samples)  # actual cost

        # Loop over text in continuatinos and extract results
        data["predicted"] = []
        errors = []
        for text in continuations:
            error_type_pred, reason_pred, _ = prompt.get_result(text)
            errors.append(error_type_pred)
            data["predicted"].append({"error": error_type_pred, "reason": reason_pred})

        # Run majority voting given the predicted error tpes
        counts = Counter(errors)
        final_prediction, count = counts.most_common()[0]

        outcome = "incorrect"
        if final_prediction == error_type:
            print(f"Got example {i} correct ({count}/{len(errors)})")
            outcome = "correct"
        data["outcome"] = outcome
        output.append(data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    parser.add_argument("--test_json", type=str,default="./config/test.json", help="Json file containing examples with labels")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--output_json", type=str, default="./data/results.json", help="path to output json to store results")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    get_accuracy(args.test_json, args.prompt_config_path, args.output_json, api_key=args.api_key)


if __name__ == "__main__":
    main()
