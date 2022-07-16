import argparse

from mer.lm import LanguageModel
from mer.prompt import Prompt


def get_results_dbls(ref_dbl, rec_dbl, prompt_config_path):
    ref_list = ref_dbl.read().split("\n")
    rec_list = rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt = Prompt.from_file(prompt_config_path)
    lm = LanguageModel()

    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file) as ref_h, open(rec_file) as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        prompt_string = prompt.create_prompt(ref, rec)
        lm.print_cost(prompt_string)
        break
        text, _ = lm.get_continuation(prompt_string)
        error_type, reason, score = prompt.get_result(text)

        print(error_type)
        print(reason)
        print(score)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dbl",
        type=argparse.FileType("r"),
        required=True,
        help="Dbl file containing paths to reference transcripts",
    )
    parser.add_argument(
        "--rec_dbl",
        type=argparse.FileType("r"),
        required=True,
        help="Dbl file containing paths to recognised transcripts",
    )
    parser.add_argument(
        "--prompt_config_path",
        type=str,
        default="./config/prompt.json",
        help="path to prompt config json",
    )
    args = parser.parse_args()

    get_results_dbls(args.ref_dbl, args.rec_dbl, args.prompt_config_path)


if __name__ == "__main__":
    main()
