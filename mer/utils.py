import json
import pandas
import re
from collections import Counter

from kaldialign import align  # pylint: disable=E0611

GAP = "***"


def convert_err_str_to_penalty(err_str):
    """
    param: err_str: string of error codes (m = minor, s = standard, e = serious)
    returns: float representing total penalty (m = 0.25, s = 0.5, e = 1.0)
    """
    err_map = {"m": 0.25, "s": 0.5, "e": 1.0}

    penalty = sum([err_map[char] for char in err_str])
    return penalty


def convert_err_str_to_dict(err_str):
    """
    param: err_str: string of error codes (m = minor, s = standard, e = serious)
    returns: readable dict of errors
    """
    err_map = {"m": "minor", "s": "standard", "e": "serious"}
    full_err_str = [err_map[err] for err in err_str]
    return " ".join(full_err_str)


def majority_voting(continuations, prompt):
    # Loop over text in continuations and extract results
    predictions = []
    err_strings = []
    for text in continuations:
        error_counts_dict, err_str = prompt.get_result(text)
        err_strings.append(err_str)
        predictions.append(error_counts_dict)

    # Run majority voting given the predicted error tpes
    counts = Counter(err_strings)
    voted_err, vote_count = counts.most_common()[0]
    if voted_err is not None:
        penalty = convert_err_str_to_penalty(voted_err)
        voted_err = convert_err_str_to_dict(voted_err)
    else:
        penalty = None

    result = {"predictions": predictions, "voted_penalty": penalty, "voted_error": voted_err, "vote_count": vote_count}

    return penalty, result


def get_alignment(ref_text, rec_text, remove_punc=False):
    # separate punctuation and split into words
    # TODO this will fail for abbreviations e.g. Mr.

    if remove_punc:
        # Find indices of punctuation in reference text
        ref_words = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
        ref_punctuation_dict = {i: v for i, v in enumerate(ref_words) if v in ".!?,"}

        # Remove all punctuation
        ref_text = re.sub(r"[^\w\s]", "", ref_text)
        rec_text = re.sub(r"[^\w\s]", "", rec_text)
    else:
        # add spaces in front of punctuation
        ref_text = ref_text.replace(".", " .").replace("?", " ?").replace("!", " !").replace(",", " ,")
        rec_text = rec_text.replace(".", " .").replace("?", " ?").replace("!", " !").replace(",", " ,")
        ref_punctuation_dict = None

    ref_words = ref_text.split()
    rec_words = rec_text.split()

    alignment = align(ref_words, rec_words, GAP)

    reference_count = len(alignment)
    return alignment, reference_count, ref_punctuation_dict


def get_sentences(ref_text, rec_text):
    """
    We don't want to pass whole paragraphs so this splits up the reference and recognised transcript
    based on the alignment. You need the alignment to deal with the cases where a whole sentence is missing
    if the ref/rec or eos punctuation is missing.
    """
    alignment, _, punctuation_dict = get_alignment(ref_text, rec_text)

    ref_sentences = []
    rec_sentences = []
    ref_sentence = []
    rec_sentence = []
    ref_counter = 0
    for ref, rec in alignment:
        if ref != GAP:
            ref_sentence.append(ref)
            ref_counter += 1
        if rec != GAP:
            rec_sentence.append(rec)
        if ref_counter in punctuation_dict:
            if punctuation_dict[ref_counter] in ".?!":
                # Append end of sentence punctuation.
                # TODO: Do not use the ref puncutation in recognised as sometimes incorrect punctuation can change meaning.
                ref_sentence.append(punctuation_dict[ref_counter])
                rec_sentence.append(punctuation_dict[ref_counter])
                del punctuation_dict[ref_counter]
                ref_counter += 1
                # Rejoin the punctuation in the string and save to list
                ref_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(ref_sentence))
                rec_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(rec_sentence))
                ref_sentences.append(ref_sentence)
                rec_sentences.append(rec_sentence)
                # reset the current sentence
                ref_sentence = []
                rec_sentence = []
            else:
                # Append other punctuation (eg ,) to sentence and continue
                ref_sentence.append(punctuation_dict[ref_counter])
                ref_counter += 1

    return ref_sentences, rec_sentences


def calculate_wer(ref_text, rec_text):

    alignment, reference_count, _ = get_alignment(ref_text, rec_text)
    result = {"reference": ref_text, "recognised": rec_text, "reference_count": reference_count}
    if reference_count == 0:
        # reference is empty after alignment, return wer=100
        result["wer"] = "null"
        return None, reference_count, result

    comparison = [""]
    insertions, deletions, substitions = 0, 0, 0
    for (ref, rec) in alignment:
        # Correct
        if ref == rec:
            comparison.append(ref)
        # Insertion Error
        elif ref == GAP:
            insertions += 1
            # if last word was also insertion, join them together
            if "inserted" in comparison[-1]:
                prev = comparison.pop()
                prev = prev.split(" is inserted")
                prev_rec = prev[0].strip("'")
                comparison.append(f"'{prev_rec} {rec}' is inserted")
            # if last word was substitution, add it in
            elif "misrecognised" in comparison[-1]:
                prev = comparison.pop()
                prev_ref, prev_rec = prev.split(" is misrecognised as ")
                prev_rec = prev_rec.strip("'")
                comparison.append(f"{prev_ref} is misrecognised as '{rec} {prev_rec}")
            else:
                comparison.append(f"['{rec}' is inserted ]")
        # Deletion Error
        elif rec == GAP:
            deletions += 1
            # if last word was also deletion, join them together
            if "deleted" in comparison[-1]:
                prev = comparison.pop()
                prev = prev.split(" is deleted")
                prev_ref = prev[0].strip("'")
                comparison.append(f"{prev_ref} {ref}' is deleted]")
            # if last word was substitution, add it in
            elif "misrecognised" in comparison[-1]:
                prev = comparison.pop()
                prev_ref, prev_rec = prev.split(" is misrecognised as ")
                prev_ref = prev_ref.strip("'")
                comparison.append(f"{prev_ref} {ref}' is misrecognised as {prev_rec}")
            else:
                comparison.append(f"['{ref}' is deleted]")
        # Subsitution Error
        else:
            substitions += 1
            # if last word was also substitution, join them together
            if "misrecognised" in comparison[-1]:
                prev = comparison.pop()
                prev_ref, prev_rec = prev.split(" is misrecognised as ")
                prev_ref = prev_ref.strip("'")
                prev_rec = prev_rec[:-1].strip("'")
                comparison.append(f"{prev_ref} {ref}' is misrecognised as '{prev_rec} {rec}']")
            else:
                comparison.append(f"['{ref}' is misrecognised as '{rec}']")

    num_errors = insertions + deletions + substitions
    wer = 100 * num_errors / reference_count
    comparison = " ".join(comparison)

    result.update(
        [
            ("comparison", comparison),
            ("insertions", insertions),
            ("deletions", deletions),
            ("substitions", substitions),
            ("wer", round(wer, 2)),
        ]
    )

    return num_errors, reference_count, result


def save_results(
    output_json,
    results,
    total_tokens,
    cost,
    total_reference_count,
    total_penalty,
    meaning_error_rate,
    wer,
    meaning_error_rate_target=None,
):
    output = {}
    output["results"] = results
    output["usage"] = {
        "total_tokens": total_tokens,
        "cost": cost,
    }
    output["summary"] = {
        "wer": round(wer, 2),
        "total_reference_count": total_reference_count,
        "total_penalty": total_penalty,
        "meaning_error_rate": round(meaning_error_rate, 2),
    }
    if meaning_error_rate_target:
        output["summary"]["accuracy"] = round(meaning_error_rate_target, 2)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def calculate_meaning_error_rate(total_reference_count, total_penalty):
    # All serious errors makes this error rate go to 100%, no errors and it is 0%
    return 100 * total_penalty / total_reference_count


def convert_excel_to_json(excel_path="./config/NER_errors_9-3-23_modified.xlsx", json_path="./config/pablo.json"):
    excel_data_df = pandas.read_excel(excel_path, sheet_name="speechmatics")
    raw_data = json.loads(excel_data_df.to_json(orient="records"))
    data = {"examples": []}
    for item in raw_data:
        data["examples"].append(
            {
                "reference": item["reference"].lower(),
                "recognised": item["recognised"].lower(),
                "reason": item["reason"],
                "minor": item["error"].lower().count("minor"),
                "standard": item["error"].lower().count("standard"),
                "serious": item["error"].lower().count("serious"),
            }
        )

    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def convert_txt_to_dict(examples):
    """
    param: examples: string containing reference, recognised, reason (optional), and severity errors (optional); one per line
    expected formatting includes a linebreak between sentence-pairs
    return: the same data but in dictionary format
    """

    ref_rec_pairs = examples.split("\n\n")

    examples = []
    for item in ref_rec_pairs:
        example = {"minor": 0, "standard": 0, "serious": 0}
        for line in item.split("\n"):
            key, value = line.split(":")
            key = key.lower()
            value = value.strip()
            # severity errors need to be split into minor, standard, serious
            if key == "error":
                for word in value.split():
                    example[word] += 1
            # everything else goes in as is
            elif key in ["reference", "recognised", "reason"]:
                example[key] = value
            else:
                exit(f"Text file contains an unrecognised keyword: {key}")

        examples.append(example)

    return examples
