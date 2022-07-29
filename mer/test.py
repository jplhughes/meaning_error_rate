import argparse
import json

from mer.mer import get_meaning_error_rate


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

    with open(args.test_json, "r", encoding="utf-8") as f:
        testset = json.load(f)
        examples = testset["examples"]

    meaning_error_rate, accuracy = get_meaning_error_rate(
        examples, args.prompt_config_path, args.output_json, api_key=args.api_key, num_samples=args.num_samples
    )
    print(f"meaning_error_rate: {meaning_error_rate}%, accuracy: {accuracy}%")


if __name__ == "__main__":
    main()
