import json
import unittest

from mer.lm import LanguageModel
from mer.prompt import Prompt


def test_results():
    config_path = "./config/prompt.json"
    with open(config_path, "r", encoding="utf-8") as f:
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


def test_api():
    with open("./unittests/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    lm = LanguageModel()
    continuations, _ = lm.get_continuation(prompt)
    assert "Jupiter" in continuations[0]


if __name__ == "__main__":
    unittest.main()
