import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


class LanguageModel:
    def __init__(self, model="text-davinci-002"):
        self.model = model

    def get_continuation(
        self,
        prompt,
        temperature=0.7,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        item = response["choices"][0]
        assert item["finish_reason"] in ["stop", "length"]

        text = item["text"].strip()
        assert text is not None

        return text, response
