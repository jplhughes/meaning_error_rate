import json
import re
from collections import Counter

from kaldialign import align  # pylint: disable=E0611


def majority_voting(continuations, prompt):
    # Loop over text in continuatinos and extract results
    predictions = []
    errors = []
    for text in continuations:
        error_type_pred, reason_pred, _ = prompt.get_result(text)
        errors.append(error_type_pred)
        predictions.append({"error": error_type_pred, "reason": reason_pred})

    # Run majority voting given the predicted error tpes
    counts = Counter(errors)
    voted_prediction, vote_count = counts.most_common()[0]

    result = {
        "predictions": predictions,
        "voted_prediction": voted_prediction,
        "vote_count": vote_count,
    }

    return voted_prediction, vote_count, result


def calculate_wer(ref_text, rec_text):
    GAP = "***"
    # separate punctuation and split into words
    # TODO this will fail for abbreviations e.g. Mr.
    ref_words = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
    rec_words = re.findall(r"[\w'-]+|[.,!?;]", rec_text)

    edits = align(ref_words, rec_words, GAP)

    comparison = ["Key: [recognised reference] {deletion} <insertion>\n"]
    insertions, deletions, substitions = 0, 0, 0
    for (ref, rec) in edits:
        # Correct
        if ref == rec:
            comparison.append(ref)
        # Insertion Error
        elif ref == GAP:
            insertions += 1
            comparison.append(f"<{rec}>")
        # Deletion Error
        elif rec == GAP:
            deletions += 1
            comparison.append("{" + ref + "}")
        # Subsitution Error
        else:
            substitions += 1
            comparison.append(f"[{rec} {ref}]")

    reference_count = len(ref_words)
    num_errors = insertions + deletions + substitions
    wer = 100 * num_errors / reference_count
    comparison = " ".join(comparison)

    result = {
        "reference": ref_text,
        "recognised": rec_text,
        "comparison": comparison,
        "insertions": insertions,
        "deletions": deletions,
        "substitions": substitions,
        "reference_count": reference_count,
        "wer": round(wer, 2),
    }

    return num_errors, reference_count, result


def save_results(
    output_json, results, total_tokens, cost, total_num_sentences, total_penalty, meaning_error_rate, wer, accuracy=None
):
    output = {}
    output["results"] = results
    output["usage"] = {
        "total_tokens": total_tokens,
        "cost": cost,
    }
    output["summary"] = {
        "wer": round(wer, 2),
        "total_num_sentences": total_num_sentences,
        "total_penalty": total_penalty,
        "meaning_error_rate": round(meaning_error_rate, 2),
    }
    if accuracy:
        output["summary"]["accuracy"] = round(accuracy, 2)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def calculate_meaning_error_rate(total_num_sentences, total_penalty):
    # All serious errors makes this accuracy go to 0%, no errors and it is 100%
    meaning_accuracy = 100 * (total_num_sentences - total_penalty) / total_num_sentences
    return 100 - meaning_accuracy
