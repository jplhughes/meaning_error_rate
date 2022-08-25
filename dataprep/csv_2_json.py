import argparse
import csv
import json
import sys
from collections import defaultdict

from mer.utils import calculate_wer, get_sentences


def csv_2_json(csv_path, json_path):
    csv_reader = csv.DictReader(csv_path, delimiter=",")
    sentences_dict = defaultdict(list)
    for row in csv_reader:
        # get reference and recognised transcripts, align and split them on a per sentence basis
        ref_text, rec_text = get_sentences(row["content"], row["amazon_transcription"])
        for ref, rec in zip(ref_text, rec_text):
            wer_results = calculate_wer(ref, rec)
            if wer_results[1] == 0 or rec in ".?!":
                # This means the reference or recognised is empty. Ignore these test cases.
                continue
            sentences_dict["examples"].append(
                {
                    "reference": ref,
                    "recognised": rec,
                    "mimir": wer_results[2]["comparison"],
                    "minor": "",
                    "standard": "",
                    "serious": "",
                }
            )

    json.dump(sentences_dict, json_path, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=argparse.FileType("r"))
    parser.add_argument("--json_out_path", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args()
    csv_2_json(args.csv_path, args.json_out_path)
