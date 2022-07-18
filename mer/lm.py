import os

import openai

# Cost for 1k tokens for each model
models2cost = {
    "text-davinci-002": 0.0600,
    "text-curie-001": 0.0060,
    "text-babbage-001": 0.0012,
    "text-ada-001": 0.0008,
}


class LanguageModel:
    """
    LM object that handles the open AI API and response given a prompt.
    """

    def __init__(self, model="text-davinci-002", api_key=None):
        self.model = model
        assert model in models2cost, f"Model {model} not supported"

        # Use api key passed in or environment variable if not
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key != "", "Pass api_key or set OPENAI_API_KEY evironment variable"
        openai.api_key = self.api_key

    def get_continuation(self, prompt, temperature=0.7, max_tokens=64, num_samples=1):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            best_of=num_samples,
            n=num_samples,
        )

        continuations = []
        for item in response["choices"]:
            text = item["text"].strip()
            assert text is not None
            continuations.append(item["text"].strip())

        return continuations, response

    def print_actual_cost(self, tokens):
        cost = models2cost[self.model] * tokens / 1000
        print(f"COST: #tokens: {tokens}, cost: ${cost:.2f}, runs/$: {1/cost:.1f}")
        return cost

    def print_estimated_cost(self, prompt, num_samples=1):
        # Find estimate number of tokens based on prompt
        approx_completion_tokens = 25 * num_samples  # based on ~100 char continuation, ~20 words
        approx_prompt_tokens = len(prompt) / 4  # each token is approximately 4 chars
        tokens = approx_prompt_tokens + approx_completion_tokens
        cost = models2cost[self.model] * tokens / 1000
        print(f"COST: #char: {len(prompt)}, #tokens: {tokens}, cost: ${cost:.2f}, runs/$: {1/cost:.1f}")
