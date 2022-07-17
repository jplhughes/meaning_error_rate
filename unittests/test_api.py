import unittest

from mer.lm import LanguageModel


class TestAPI(unittest.TestCase):
    def test_api(self):
        with open("./unittests/prompt.txt", "r") as f:
            prompt = f.read().strip()
        lm = LanguageModel()
        text, _ = lm.get_continuation(prompt)
        self.assertIn("Jupiter", text)


if __name__ == "__main__":
    unittest.main()
