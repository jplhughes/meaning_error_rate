import json
from collections import Counter


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

    return voted_prediction, vote_count, predictions


def create_result_dict(ref, rec, predictions, voted_prediction, vote_count):
    result = {"reference": ref, "recognised": rec}
    result["predictions"] = predictions
    result["voted_prediction"] = voted_prediction
    result["vote_count"] = vote_count
    return result


def save_results(
    output_json, results, total_tokens, cost, total_num_sentences, total_penalty, meaning_error_rate, accuracy=None
):
    output = {}
    output["results"] = results
    output["usage"] = {
        "total_tokens": total_tokens,
        "cost": cost,
    }
    output["summary"] = {
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
