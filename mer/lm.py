import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Cost for 1k tokens for each model
models2cost = {
    "text-davinci-002": 0.0600,
    "text-curie-001": 0.0060,
    "text-babbage-001": 0.0012,
    "text-ada-001": 0.0008,
}


class LanguageModel:
    def __init__(self, model="text-davinci-002"):
        self.model = model
        assert model in models2cost, f"Model {model} not supported"

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

    def print_cost(self, prompt):
        tokens = len(prompt) / 4  # each token is approximately 4 chars
        cost = models2cost[self.model] * tokens / 1000
        print(f"#char: {len(prompt)}, #token: {tokens}, cost: ${cost}, runs/$: {1/cost:.1f}")
