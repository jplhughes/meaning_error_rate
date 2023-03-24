from mer.lm import LanguageModel
from mer.prompt import PromptMultiple
from mer.utils import (
    calculate_meaning_error_rate,
    calculate_wer,
    majority_voting,
    save_results,
)


def get_meaning_error_rate(
    examples, prompt_config_path, output_json, api_key=None, num_samples=3, simple=False, dry_run=False
):
    prompt = PromptMultiple.from_file(prompt_config_path, simple=simple)
    lm = LanguageModel(api_key=api_key)

    total_errors, total_reference_count = 0, 0  # For WER
    total_penalty, total_target_penalty, total_tokens = 0, 0, 0  # For MER
    cost = 0

    results = []
    for i, example in enumerate(examples):
        error_count_target, ref, rec = prompt.unpack_example(example)

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
        voted_penalty, prediction_result = majority_voting(continuations, prompt)
        mer_pred = calculate_meaning_error_rate(reference_count, voted_penalty)
        prediction_result["meaning_error_rate"] = round(mer_pred, 2)
        total_penalty += voted_penalty

        # If you have human labels (targets counts for error type), then record extra stats
        if error_count_target:
            penalty_target = prompt.get_penalty(error_count_target)
            total_target_penalty += penalty_target
            mer_target = calculate_meaning_error_rate(reference_count, penalty_target)
            error_count_target["meaning_error_rate"] = round(mer_target, 2)
            prediction_result["target"] = error_count_target
            prediction_result["mer_diff"] = round(mer_pred - mer_target, 2)

        results.append({**wer_result, **prediction_result})

    if dry_run:
        print(f"This run would cost approximately: £{cost}")
        return None, None

    cost = lm.print_actual_cost(total_tokens)
    meaning_error_rate = calculate_meaning_error_rate(total_reference_count, total_penalty)
    wer = 100 * total_errors / total_reference_count

    if total_target_penalty > 0:
        meaning_error_rate_target = calculate_meaning_error_rate(total_reference_count, total_target_penalty)

    save_results(
        output_json,
        results,
        total_tokens,
        cost,
        total_reference_count,
        total_penalty,
        meaning_error_rate,
        wer,
        meaning_error_rate_target,
    )

    return meaning_error_rate, meaning_error_rate_target
