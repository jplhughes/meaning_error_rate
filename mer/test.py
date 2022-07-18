import argparse
import json

from mer.lm import LanguageModel
from mer.prompt import Prompt
from mer.utils import (
    calculate_meaning_error_rate,
    create_result_dict,
    majority_voting,
    save_results,
)


def get_accuracy(test_json, prompt_config_path, output_json, api_key=None, num_samples=3, simple=False):

    with open(test_json, "r", encoding="utf-8") as f:
        testset = json.load(f)

    prompt = Prompt.from_file(prompt_config_path, simple=simple)
    lm = LanguageModel(api_key=api_key)

    total_num_sentences, total_penalty, total_correct, total_tokens = 0, 0, 0, 0
    results = []
    for i, example in enumerate(testset["examples"]):
        error_type, ref, rec, reason = prompt.unpack_example(example)

        # Create prompt and get continuations from lm
        prompt_string = prompt.create_prompt(ref, rec)
        continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
        total_tokens += response["usage"]["total_tokens"]

        # Get prediction for each continuation and find the most common error
        voted_prediction, vote_count, predictions = majority_voting(continuations, prompt)

        outcome = "incorrect"
        if voted_prediction == error_type:
            print(f"Got example {i} correct ({vote_count}/{len(continuations)})")
            outcome = "correct"
            total_correct += 1

        # Add example information to item in json
        result = create_result_dict(ref, rec, predictions, voted_prediction, vote_count)
        # Add label specific info
        result["target"] = {"error": error_type, "reason": reason}
        result["outcome"] = outcome
        results.append(result)

        # Keep track of score penalties to work out MER
        total_num_sentences += 1
        total_penalty += prompt.error2score[voted_prediction]

    cost = lm.print_actual_cost(total_tokens)
    meaning_error_rate = calculate_meaning_error_rate(total_num_sentences, total_penalty)

    # Accuracy of LLM method to match human labels
    accuracy = 100 * total_correct / total_num_sentences

    save_results(
        output_json, results, total_tokens, cost, total_num_sentences, total_penalty, meaning_error_rate, accuracy
    )

    return accuracy, meaning_error_rate


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    parser.add_argument("--test_json", type=str,default="./config/test.json", help="Json file containing examples with labels")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--output_json", type=str, default="./results.json", help="path to output json to store results")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    parser.add_argument("--num_samples", type=str, default=3, help="number of times to sample GPT3 for majority voting")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    accuracy, meaning_error_rate = get_accuracy(
        args.test_json, args.prompt_config_path, args.output_json, api_key=args.api_key, num_samples=args.num_samples
    )
    print(f"accuracy: {accuracy}%, meaning_error_rate: {meaning_error_rate}%")


if __name__ == "__main__":
    main()
