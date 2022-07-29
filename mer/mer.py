from mer.lm import LanguageModel
from mer.prompt import Prompt
from mer.utils import (
    calculate_meaning_error_rate,
    calculate_wer,
    majority_voting,
    save_results,
)


def get_meaning_error_rate(
    examples, prompt_config_path, output_json, api_key=None, num_samples=3, simple=False, dry_run=False
):
    prompt = Prompt.from_file(prompt_config_path, simple=simple)
    lm = LanguageModel(api_key=api_key)

    total_errors, total_reference_count = 0, 0  # For WER
    total_num_sentences, total_penalty, total_tokens = 0, 0, 0  # For MER
    total_correct, total_labelled = 0, 0  # For accuracy
    cost = 0

    results = []
    for i, example in enumerate(examples):
        error_type, ref, rec, reason = prompt.unpack_example(example)

        # WER
        errors, reference_count, wer_result = calculate_wer(ref, rec)
        total_errors += errors
        total_reference_count += reference_count

        # Create prompt and get continuations from lm
        prompt_string = prompt.create_prompt(ref, rec)
        if dry_run:
            cost += lm.print_estimated_cost(prompt_string, num_samples=num_samples)  # estimated cost
            print(i, prompt_string)
            continue
        continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
        total_tokens += response["usage"]["total_tokens"]

        # Majority voting (keep track of score penalties to work out MER)
        voted_prediction, vote_count, prediction_result = majority_voting(continuations, prompt)
        total_num_sentences += 1
        total_penalty += prompt.error2score[voted_prediction]

        # If you have human labels (targets for error type), then record extra stats
        if error_type:
            outcome = "incorrect"
            total_labelled += 1
            if voted_prediction == error_type:
                print(f"Got example {i} correct ({vote_count}/{len(continuations)})")
                outcome = "correct"
                total_correct += 1

            # Add label specific info
            prediction_result["target"] = {"error": error_type, "reason": reason}
            prediction_result["outcome"] = outcome

        results.append({**wer_result, **prediction_result})

    if dry_run:
        print(f"This run would cost approximately: £{cost}")
        return None, None

    cost = lm.print_actual_cost(total_tokens)
    meaning_error_rate = calculate_meaning_error_rate(total_num_sentences, total_penalty)
    wer = 100 * total_errors / total_reference_count

    # Accuracy of LLM method to match human labels (if they were provided)
    if total_labelled > 0:
        accuracy = 100 * total_correct / total_labelled
    else:
        accuracy = None

    save_results(
        output_json, results, total_tokens, cost, total_num_sentences, total_penalty, meaning_error_rate, wer, accuracy
    )

    return meaning_error_rate, accuracy
