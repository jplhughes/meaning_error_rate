import os

import openai
from tenacity import retry, stop_after_attempt, wait_fixed

complete_models = ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"]
chat_models = ["gpt-3.5-turbo", "gpt-4"]
# Cost for 1k tokens for each model
models2cost = {
    "gpt-3.5-turbo": 0.0020,
    "text-davinci-003": 0.0200,
    "text-davinci-002": 0.0200,
    "text-curie-001": 0.0020,
    "text-babbage-001": 0.0005,
    "text-ada-001": 0.0004,
    "gpt-4": 0.0600
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

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(5))
    def get_continuation(self, prompt, temperature=0.7, max_tokens=512, num_samples=1):
        if self.model in complete_models:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                best_of=num_samples,
                n=num_samples,
            )
        elif self.model in chat_models:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for error classification."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=num_samples
            )
        else:
            raise ValueError(f"Model {self.model} is not supported")

        continuations = []
        for item in response["choices"]:
            if self.model in complete_models:
                text = item["text"].strip()
            elif self.model in chat_models:
                text = item["message"]["content"].strip()
            else:
                raise ValueError(f"Model {self.model} is not supported")
            assert text is not None
            continuations.append(text)

        return continuations, response

    def print_actual_cost(self, tokens):
        cost = models2cost[self.model] * tokens / 1000
        print(f"COST: #tokens: {tokens}, cost: ${cost:.2f}, runs/$: {1/cost:.1f}")
        return round(cost, 2)

    def print_estimated_cost(self, prompt, num_samples=1):
        # Find estimate number of tokens based on prompt
        approx_completion_tokens = 25 * num_samples  # based on ~100 char continuation, ~20 words
        approx_prompt_tokens = len(prompt) / 4  # each token is approximately 4 chars
        tokens = approx_prompt_tokens + approx_completion_tokens
        cost = models2cost[self.model] * tokens / 1000
        print(f"COST: #char: {len(prompt)}, #tokens: {tokens}, cost: ${cost:.2f}, runs/$: {1/cost:.1f}")
        return round(cost, 2)
