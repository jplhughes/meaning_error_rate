import argparse

from mer.mer import get_meaning_error_rate


def convert_dbl_to_dict(ref_dbl, rec_dbl):
    ref_list = ref_dbl.read().split("\n")
    rec_list = rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    examples = []
    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file, "r", encoding="utf-8") as ref_h, open(rec_file, "r", encoding="utf-8") as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

            example = {
                "reference": ref,
                "recognised": rec,
            }
            examples.append(example)

    return examples


def convert_txt_to_dict(txt):
    examples = txt.read()
    ref_rec_pairs = examples.split("\n\n")

    examples = []
    for item in ref_rec_pairs:
        example = {"minor": 0, "standard": 0, "serious": 0}
        for line in item.split("\n"):
            key, value = line.split(":")
            key = key.lower()
            if key == "target":
                for word in value:
                    example[word] += 1
            elif key in ["reference", "recognised", "reason"]:
                example[key] = value
            else:
                exit(f"Text file contains an unrecognised keyword: {key}")

        examples.append(example)

    return examples


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--example_path", type=argparse.FileType("r"), required=False, help="txt file containing reference, recognised pairs for processing")
    dbls = inputs.add_argument_group()
    dbls.add_argument("--ref_dbl", type=argparse.FileType("r"), required=False, help="Dbl file containing paths to reference transcripts")  # noqa:  E201
    dbls.add_argument("--rec_dbl", type=argparse.FileType("r"), required=False, help="Dbl file containing paths to recognised transcripts")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--output_json", type=str, default="./results_dbl.json", help="path to output json to store results")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    parser.add_argument("--num_samples", type=str, default=3, help="number of times to sample GPT3 for majority voting")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    if args.example_path:
        examples = convert_txt_to_dict(args.example_path)
    else:
        examples = convert_dbl_to_dict(args.ref_dbl, args.rec_dbl)

    meaning_error_rate, _ = get_meaning_error_rate(
        examples, args.prompt_config_path, args.output_json, api_key=args.api_key, num_samples=args.num_samples
    )
    print(f"meaning_error_rate: {meaning_error_rate}%")


if __name__ == "__main__":
    main()
