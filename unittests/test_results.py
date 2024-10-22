import json
import unittest

from mer.lm import LanguageModel
from mer.mer import get_meaning_error_rate
from mer.prompt import Prompt
from mer.run import convert_dbl_to_dict


def test_api():
    with open("./unittests/data/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    lm = LanguageModel()
    continuations, _ = lm.get_continuation(prompt)
    assert "Jupiter" in continuations[0]


def test_expected_error_received():
    prompt_config_path = "./config/prompt.json"
    with open(prompt_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Pick example to remove from prompt and use that to test on
    test_example = config["examples"].pop(0)
    prompt = Prompt(config, simple=True)
    lm = LanguageModel()

    _, ref, rec, _ = prompt.unpack_example(test_example)
    prompt_string = prompt.create_prompt(ref, rec)
    continuations, _ = lm.get_continuation(prompt_string)
    error_type_pred, _, _ = prompt.get_result(continuations[0])

    assert error_type_pred in list(prompt.error2score)


def test_running_with_dbls():
    prompt_config_path = "./unittests/data/prompt.json"
    ref_dbl = "./unittests/data/ref.dbl"
    rec_dbl = "./unittests/data/rec.dbl"
    output_json = "./unittests/data/results_dbl.json"
    with open(ref_dbl, "r", encoding="utf-8") as ref, open(rec_dbl, "r", encoding="utf-8") as rec:
        examples = convert_dbl_to_dict(ref, rec)
    meaning_error_rate, _ = get_meaning_error_rate(
        examples, prompt_config_path, output_json, num_samples=2, simple=True
    )
    print(f"meaning_error_rate: {meaning_error_rate}%")


def test_running_with_testset():
    prompt_config_path = "./unittests/data/prompt.json"
    test_json = "./unittests/data/test.json"
    output_json = "./unittests/data/results.json"
    with open(test_json, "r", encoding="utf-8") as f:
        testset = json.load(f)
        examples = testset["examples"]
    meaning_error_rate, accuracy = get_meaning_error_rate(
        examples, prompt_config_path, output_json, num_samples=2, simple=False
    )
    print(f"meaning_error_rate: {meaning_error_rate}%, accuracy: {accuracy}%")


if __name__ == "__main__":
    unittest.main()
