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
