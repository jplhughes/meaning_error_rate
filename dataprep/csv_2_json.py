import argparse
import csv
import json
from collections import defaultdict

from mer.utils import calculate_wer, get_sentences

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str)
parser.add_argument("--json_out_path", type=str)


def csv_2_json(csv_path, json_path):
    with open(csv_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        counter = 0
        sentences_dict = defaultdict(list)
        for i, row in enumerate(csv_reader):
            if counter == 0 or counter == 1:
                counter += 1
                continue
            else:
                ref_text, rec_text = get_sentences(row[7], row[12])
                for ref, rec in zip(ref_text, rec_text):
                    wer_results = calculate_wer(ref, rec)
                    if wer_results is None:
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

        with open(json_path, "w") as outfile:
            json.dump(sentences_dict, outfile, indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    csv_2_json(args.csv_path, args.json_out_path)
