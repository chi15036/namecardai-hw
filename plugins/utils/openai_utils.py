import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry


class OpenaiUtils:
    def __init__(
        self,
        embedding_model="text-embedding-3-small",
        completion_model="gpt-4o-mini",
    ):
        organization = os.environ["OPENAI_ORGANIZATION"]
        api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(
            organization=organization,
            api_key=api_key,
        )
        self.embedding_model = embedding_model
        self.completion_model = completion_model

    @sleep_and_retry
    @limits(calls=5, period=1)
    def get_embedding(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts, model=self.embedding_model
        )
        embedding = response.data[0].embedding
        return embedding

    @sleep_and_retry
    @limits(calls=5, period=1)
    def get_completion(self, messages: list):
        response = self.client.chat.completions.create(
            model=self.completion_model,
            seed=123,
            messages=messages,
        )
        return dict(response.choices[0].message)["content"]

    def mget_completion(self, messages: list) -> list:
        completions = []
        futures = []
        with ThreadPoolExecutor() as executor:
            for message in messages:
                future = executor.submit(self.get_completion, message)
                futures.append(future)
            for future in as_completed(futures):
                completions.append(future.result())
        return completions
