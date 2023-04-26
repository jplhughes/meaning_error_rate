import argparse
import json

from mer.mer import get_meaning_error_rate
from mer.utils import convert_txt_to_dict


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    parser.add_argument("--test_file", type=str,default="./config/comparison_prompt.txt", help="Json or txt file containing examples with labels")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/comparison_prompt_base.txt", help="path to prompt config json or txt file")  # noqa:  E201
    parser.add_argument("--output_json", type=str, default="./results.json", help="path to output json to store results")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    parser.add_argument("--num_samples", type=str, default=3, help="number of times to sample GPT3 for majority voting")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    if args.test_file.endswith(".txt"):
        with open(args.test_file) as f:
            testset = f.read()
        examples = convert_txt_to_dict(testset)
    else:
        with open(args.test_file, "r", encoding="utf-8") as f:
            testset = json.load(f)
        examples = testset["examples"]

    meaning_error_rate, accuracy = get_meaning_error_rate(
        examples, args.prompt_config_path, args.output_json, api_key=args.api_key, num_samples=args.num_samples
    )
    print(f"predicted_meaning_error_rate: {meaning_error_rate}%, actual_meaning_error_rate: {accuracy}%")


if __name__ == "__main__":
    main()
