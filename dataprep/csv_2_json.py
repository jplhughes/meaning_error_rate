import argparse
import csv
import json
from collections import defaultdict

from mer.utils import get_sentences

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str)
parser.add_argument("--json_out_path", type=str)


def csv_2_json(csv_path, json_path):
    with open(csv_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        counter = 0
        sentences_dict = defaultdict(list)
        for row in csv_reader:
            if counter == 0:
                counter += 1
                continue
            else:
                ref_text, rec_text = get_sentences(row[7], row[8])
                for ref, rec in zip(ref_text, rec_text):
                    sentences_dict["examples"].append({"reference": ref, "recognised": rec})

        with open(json_path, "w") as outfile:
            json.dump(sentences_dict, outfile, indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    csv_2_json(args.csv_path, args.json_out_path)