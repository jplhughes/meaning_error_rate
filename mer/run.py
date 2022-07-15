import argparse

from mer.utils import Prompt


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

    ref_list = args.ref_dbl.read().split("\n")
    rec_list = args.rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt = Prompt(args.prompt_config_path)

    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file) as ref_h, open(rec_file) as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        prompt_string = prompt.create_prompt(ref, rec)

        print(prompt_string)


if __name__ == "__main__":
    main()
