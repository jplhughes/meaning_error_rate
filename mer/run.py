import argparse

from mer.lm import LanguageModel
from mer.prompt import Prompt
from mer.utils import (
    calculate_meaning_error_rate,
    calculate_wer,
    majority_voting,
    save_results,
)


def get_results_dbls(
    ref_dbl, rec_dbl, prompt_config_path, output_json, api_key=None, num_samples=3, simple=False, dry_run=False
):
    ref_list = ref_dbl.read().split("\n")
    rec_list = rec_dbl.read().split("\n")

    assert len(ref_list) == len(rec_list), "Length of reference and recognised dbls differ"

    prompt = Prompt.from_file(prompt_config_path, simple=simple)
    lm = LanguageModel(api_key=api_key)

    total_num_sentences, total_penalty, total_tokens = 0, 0, 0
    total_errors, total_reference_count = 0, 0
    results = []
    for ref_file, rec_file in zip(ref_list, rec_list):
        with open(ref_file, "r", encoding="utf-8") as ref_h, open(rec_file, "r", encoding="utf-8") as rec_h:
            ref = ref_h.read().strip()
            rec = rec_h.read().strip()

        # WER
        errors, reference_count, wer_result = calculate_wer(ref, rec)
        total_errors += errors
        total_reference_count += reference_count

        # Prompt and LM continuation
        prompt_string = prompt.create_prompt(ref, rec)
        if dry_run:
            lm.print_estimated_cost(prompt_string, num_samples=num_samples)  # estimated cost
            print(prompt_string)
            continue

        continuations, response = lm.get_continuation(prompt_string, num_samples=num_samples)
        total_tokens += response["usage"]["total_tokens"]

        # Majority voting (keep track of score penalties to work out MER)
        voted_prediction, _, prediction_result = majority_voting(continuations, prompt)
        total_num_sentences += 1
        total_penalty += prompt.error2score[voted_prediction]

        results.append({**wer_result, **prediction_result})

    cost = lm.print_actual_cost(total_tokens)
    meaning_error_rate = calculate_meaning_error_rate(total_num_sentences, total_penalty)
    wer = 100 * total_errors / total_reference_count
    save_results(output_json, results, total_tokens, cost, total_num_sentences, total_penalty, meaning_error_rate, wer)

    return meaning_error_rate


def main():

    parser = argparse.ArgumentParser()
    # pylint: disable=line-too-long
    # fmt: off
    parser.add_argument("--ref_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to reference transcripts")  # noqa:  E201
    parser.add_argument("--rec_dbl", type=argparse.FileType("r"), required=True, help="Dbl file containing paths to recognised transcripts")  # noqa:  E201
    parser.add_argument("--prompt_config_path", type=str, default="./config/prompt.json", help="path to prompt config json")  # noqa:  E201
    parser.add_argument("--output_json", type=str, default="./results_dbl.json", help="path to output json to store results")  # noqa:  E201
    parser.add_argument("--api_key", type=str, default=None, help="api key for open ai")  # noqa:  E201
    parser.add_argument("--num_samples", type=str, default=3, help="number of times to sample GPT3 for majority voting")  # noqa:  E201
    # fmt: on
    args = parser.parse_args()

    meaning_error_rate = get_results_dbls(
        args.ref_dbl,
        args.rec_dbl,
        args.prompt_config_path,
        args.output_json,
        api_key=args.api_key,
        num_samples=args.num_samples,
    )
    print(f"meaning_error_rate: {meaning_error_rate}%")


if __name__ == "__main__":
    main()
