import re

from kaldialign import align  # pylint: disable=E0611

GAP = "***"


def calculate_changes(ref_words, rec_words):

    # separate punctuation and split into words
    ref_words = re.findall(r"[\w']+|[.,!?;]", ref_words)
    rec_words = re.findall(r"[\w']+|[.,!?;]", rec_words)

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
            comparison.append(f"{{ref}}")
        # Subsitution Error
        else:
            substitions += 1
            comparison.append(f"[{rec} {ref}]")

    reference_count = len(ref_words)
    wer = (insertions + deletions + substitions) / reference_count
    comparison = " ".join(comparison)

    item = {
        "comparison": comparison,
        "insertions": insertions,
        "deletions": deletions,
        "substitions": substitions,
        "reference_count": reference_count,
        "wer": wer,
    }

    return item
