import os
import unittest

import openai


class Test_API(unittest.TestCase):
    def test_api(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assertIsNotNone(openai.api_key)

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt="Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",  # noqa: E501
            temperature=0.7,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        item = response["choices"][0]
        self.assertIn(item["finish_reason"], ["stop", "length"])

        text = item["text"].strip().replace("\n", "")
        self.assertIsNotNone(text)
        self.assertIn("Jupiter", text)


if __name__ == "__main__":
    unittest.main()
