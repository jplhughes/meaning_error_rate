import json
import re
from collections import Counter

from kaldialign import align  # pylint: disable=E0611

GAP = "***"


def majority_voting(continuations, prompt):
    # Loop over text in continuatinos and extract results
    predictions = []
    penalities = []
    for text in continuations:
        error_counts_dict, penalty = prompt.get_result(text)
        penalities.append(penalty)
        predictions.append(error_counts_dict)

    # Run majority voting given the predicted error tpes
    counts = Counter(penalities)
    voted_penality, vote_count = counts.most_common()[0]

    result = {
        "predictions": predictions,
        "voted_penality": voted_penality,
        "vote_count": vote_count,
    }

    return voted_penality, result


def get_alignment(ref_text, rec_text):
    # separate punctuation and split into words
    # TODO this will fail for abbreviations e.g. Mr.
    ref_words = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
    rec_words = re.findall(r"[\w'-]+|[.,!?;]", rec_text)

    alignment = align(ref_words, rec_words, GAP)
    reference_count = len(ref_words)
    return alignment, reference_count


def get_sentences(ref_text, rec_text):
    """
    We don't want to pass whole paragraphs so this splits up the reference and recognised transcript
    based on the alignment. You need the alignment to deal with the cases where a whole sentence is missing
    if the ref/rec or eos punctuation is missing.
    """
    alignment, _ = get_alignment(ref_text, rec_text)

    ref_sentences = []
    rec_sentences = []
    ref_sentence = []
    rec_sentence = []
    for (ref, rec) in alignment:
        # If EOS found in reference, start a new sentence at that point in the alignment
        if ref in ".!?":
            ref_sentence.append(ref)
            if rec != GAP:
                rec_sentence.append(rec)
            # Rejoin the punctuation in the string and save to list
            ref_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(ref_sentence))
            rec_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(rec_sentence))
            ref_sentences.append(ref_sentence)
            rec_sentences.append(rec_sentence)
            # reset the current sentence
            ref_sentence = []
            rec_sentence = []
        else:
            # Kepp adding words if they are not a "gap"
            if ref != GAP:
                ref_sentence.append(ref)
            if rec != GAP:
                rec_sentence.append(rec)
    return ref_sentences, rec_sentences


def calculate_wer(ref_text, rec_text):

    alignment, reference_count = get_alignment(ref_text, rec_text)

    comparison = ["Key: [recognised reference] {deletion} <insertion>\n"]
    insertions, deletions, substitions = 0, 0, 0
    for (ref, rec) in alignment:
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
