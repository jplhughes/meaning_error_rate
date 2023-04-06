import os
import json
from mer.lm import LanguageModel
from mer.prompt import PromptMultiple
from mer.utils import (
    calculate_meaning_error_rate,
    calculate_wer,
    majority_voting,
    save_results,
)


def get_meaning_error_rate(examples, prompt_path, output_json, api_key=None, num_samples=3, simple=False, dry_run=True):
    if prompt_path.endswith(".json"):
        prompt = PromptMultiple.from_file(prompt_path, simple=simple)
    elif prompt_path.endswith(".txt"):
        prompt = PromptMultiple.from_txt(prompt_path, simple=simple)
    else:
        exit("Prompt input type not supported")
    lm = LanguageModel(api_key=api_key)

    cost, total_tokens = 0, 0

    # if file exists, load it
    output_log = f"{output_json}.continuations.json"
    if os.path.exists(output_log):
        with open(output_log, "r", encoding="utf-8") as f:
            continuations_list = list(json.load(f))
    else:
        for example in examples:
            _, ref, rec = prompt.unpack_example(example)
            prompt_string = prompt.create_prompt(ref, rec)
            cost += lm.print_estimated_cost(prompt_string, num_samples=num_samples)

        accept_strings = ["Y", "y", "Yes", "yes"]
        if input(f"Do you want to spend ${round(cost, 2)}? Enter Y/N to continue: ") not in accept_strings:
            print("You didn't want to proceed, exiting")
            exit(1)

        continuations_list = []
        for example in examples:
            _, ref, rec = prompt.unpack_example(example)

            # Create prompt and get continuations from lm
            prompt_string = prompt.create_prompt(ref, rec)
            continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
            total_tokens += response["usage"]["total_tokens"]
            continuations_list.append(continuations)
        cost = lm.print_actual_cost(total_tokens)
        print(f"Cost: ${round(cost, 2)}")

        # save list of continuations to file
        with open(output_log, "w", encoding="utf-8") as f:
            json.dump(continuations_list, f, indent=4)

    total_errors, total_reference_count = 0, 0  # For WER
    total_penalty, total_target_penalty = 0, 0  # For MER
    results = []
    bad_examples = []
    for example, continuations in zip(examples, continuations_list):
        error_count_target, ref, rec = prompt.unpack_example(example)

        # WER
        errors, reference_count, wer_result = calculate_wer(ref, rec)
        total_errors += errors
        total_reference_count += reference_count

        # Majority voting (keep track of score penalties to work out MER)
        voted_penalty, prediction_result = majority_voting(continuations, prompt)
        if voted_penalty:
            mer_pred = calculate_meaning_error_rate(reference_count, voted_penalty)
            prediction_result["meaning_error_rate"] = round(mer_pred, 2)
            total_penalty += voted_penalty
        else:
            bad_examples.append((ref, rec, continuations))
            prediction_result["meaning_error_rate"] = None
        # If you have human labels (targets counts for error type), then record extra stats
        if error_count_target:
            penalty_target = prompt.get_penalty(error_count_target)
            total_target_penalty += penalty_target
            mer_target = calculate_meaning_error_rate(reference_count, penalty_target)
            error_count_target["meaning_error_rate"] = round(mer_target, 2)
            prediction_result["target"] = error_count_target
            prediction_result["target_mer"] = round(mer_target, 2)
            prediction_result["mer_diff"] = round(mer_pred - mer_target, 2)

        results.append({**wer_result, **prediction_result})

    meaning_error_rate = calculate_meaning_error_rate(total_reference_count, total_penalty)
    wer = 100 * total_errors / total_reference_count

    if total_target_penalty > 0:
        meaning_error_rate_target = calculate_meaning_error_rate(total_reference_count, total_target_penalty)
    else:
        meaning_error_rate_target = None

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
    with open("results/bad_examples.txt", "w") as f:
        for ref, rec, continuations in bad_examples:
            f.write(f"Reference:{ref}\n")
            f.write(f"Recognised:{rec}\n")
            f.write(f"Continuations:{continuations}\n")
            f.write("\n")
    return meaning_error_rate, meaning_error_rate_target
