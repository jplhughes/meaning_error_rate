import argparse

from mer.lm import LanguageModel
from mer.prompt import Prompt


def get_results_dbls(ref_dbl, rec_dbl, prompt_config_path, api_key=None):
    ref_list = ref_dbl.read().split("\n")
    rec_list = rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt = Prompt.from_file(prompt_config_path)
    lm = LanguageModel(api_key=api_key)

    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file) as ref_h, open(rec_file) as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        prompt_string = prompt.create_prompt(ref, rec)
        lm.print_cost(prompt_string)
        text, _ = lm.get_continuation(prompt_string)
        error_type, reason, score = prompt.get_result(text)

        print(error_type)
        print(reason)
        print(score)
        break


def main():

    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument( "--ref_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to reference transcripts")  # noqa:  E201
    parser.add_argument( "--rec_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to recognised transcripts")  # noqa:  E201
    parser.add_argument( "--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    get_results_dbls(args.ref_dbl, args.rec_dbl, args.prompt_config_path, api_key=args.api_key)


if __name__ == "__main__":
    main()
