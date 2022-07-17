import argparse

from mer.lm import LanguageModel
from mer.prompt import Prompt


def get_results_dbls(ref_dbl, rec_dbl, prompt_config_path, api_key=None, num_samples=3):
    ref_list = ref_dbl.read().split("\n")
    rec_list = rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt = Prompt.from_file(prompt_config_path)
    lm = LanguageModel(api_key=api_key)

    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file, "r", encoding="utf-8") as ref_h, open(rec_file, "r", encoding="utf-8") as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        prompt_string = prompt.create_prompt(ref, rec)
        lm.print_cost(prompt_string, num_samples=num_samples)  # estimated cost
        continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
        lm.print_cost(prompt_string, tokens=response["usage"]["total_tokens"], num_samples=num_samples)  # actual cost

        print(response)
        for text in continuations:
            error_type, reason, score = prompt.get_result(text)
            print(error_type, score, reason)

        break


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    parser.add_argument("--ref_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to reference transcripts")  # noqa:  E201
    parser.add_argument("--rec_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to recognised transcripts")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    get_results_dbls(args.ref_dbl, args.rec_dbl, args.prompt_config_path, api_key=args.api_key)


if __name__ == "__main__":
    main()
