import json
import re
from collections import Counter

from kaldialign import align  # pylint: disable=E0611

GAP = "***"


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


def get_alignment(ref_text, rec_text):
    # separate punctuation and split into words
    # TODO this will fail for abbreviations e.g. Mr.

    # Find indices of punctuation in reference text
    ref_words = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
    punctuation_dict = {}
    for i, token in enumerate(ref_words):
        if token in ".!?,":
            punctuation_dict[i] = token

    # Remove all punctuation and align
    ref_text = re.sub(r"[^\w\s]", "", ref_text)
    rec_text = re.sub(r"[^\w\s]", "", rec_text)
    ref_words = ref_text.split()
    rec_words = rec_text.split()
    alignment = align(ref_words, rec_words, GAP)

    reference_count = len(alignment)
    return alignment, reference_count, punctuation_dict


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
    for i, (ref, rec) in enumerate(alignment):
        if ref != GAP:
            ref_sentence.append(ref)
            ref_counter += 1
        if rec != GAP:
            rec_sentence.append(rec)
        if ref_counter in punctuation_dict:
            if punctuation_dict[ref_counter] in ".?!":
                # Append end of sentence punctuation.
                ref_sentence.append(punctuation_dict[ref_counter])
                rec_sentence.append(punctuation_dict[ref_counter])
                del punctuation_dict[ref_counter]
                ref_counter += 1
                # Rejoin the punctuation in the string and save to list
                ref_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(ref_sentence))
                rec_sentence = re.sub(r'\s([?.!"](?:\s|$))', r"\1", " ".join(rec_sentence))
                if rec_sentence in ".?!":
                    # if the rec_sentence is empty omit from test set
                    ref_sentence = []
                    rec_sentence = []
                    continue
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
    if reference_count == 0:
        return

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


def split_off_punc(line, punc_str):
    """
    Splits punctuation off from its words to make alignment easier
    Args:
        line (str): line to be cleaned
        punc_str (str): string containing punc marks to be separated
    Resturns:
        list: original line with punc marks removed, then split into words
    """
    out = re.sub(r"([{}]) ".format(punc_str), r" \1 ", line.strip())
    out = re.sub(r"([{}])$".format(punc_str), r" \1", out)
    return out.split()
