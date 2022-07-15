import os
import unittest

import openai

from mer.lm import LanguageModel


class Test_API(unittest.TestCase):
    def test_api(self):
        lm = LanguageModel()
        self.assertIsNotNone(openai.api_key)

        with open("./unittests/prompt.txt", "r") as f:
            prompt = f.read().strip()

        text, _ = lm.get_continuation(prompt)
        self.assertIn("Jupiter", text)


if __name__ == "__main__":
    unittest.main()
