import argparse
import copy
import json


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
        "--prompt_config",
        type=argparse.FileType("r"),
        default="./config/prompt.json",
        help="path to prompt config json",
    )
    args = parser.parse_args()

    ref_list = args.ref_dbl.read().split("\n")
    rec_list = args.rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt_config = json.load(args.prompt_config)

    prompt_base = []
    error2score = {}
    for error_type in prompt_config["errors"]:
        description = prompt_config["errors"][error_type]["description"]
        error2score[error_type] = prompt_config["errors"][error_type]["score"]

        prompt_base.append(f"{error_type.capitalize()} error - {description}.\n")

    for example in prompt_config["examples"]:
        error = example["error"]
        ref = example["reference"]
        rec = example["recognised"]
        reason = example["reason"]

        prompt_base.append(f"Reference: {ref}")
        prompt_base.append(f"Recognised: {rec}")
        prompt_base.append(f"Result: {error} due to {reason}. I hope it is correct.\n")

    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file) as ref_h, open(rec_file) as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        prompt_specific = copy.deepcopy(prompt_base)
        prompt_specific.append(f"Reference: {ref}")
        prompt_specific.append(f"Recognised: {rec}")
        prompt_specific.append(f"Result:")

        prompt = "\n".join(prompt_specific)

        print(prompt)


if __name__ == "__main__":
    main()
