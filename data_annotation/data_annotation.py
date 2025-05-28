import os
import asyncio
import time
import pandas as pd
from openai import AsyncOpenAI
from loguru import logger
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from yaml import safe_load

from loans_classification.config import LLM_API_ENDPOINT, LLM_API_AUTH_USERNAME, LLM_API_AUTH_PASSWORD
from loans_classification.data_annotation.auth import authorize_and_get_token_from_gateway
from loans_classification.models import Label

TOKEN = authorize_and_get_token_from_gateway(username=LLM_API_AUTH_USERNAME, password=LLM_API_AUTH_PASSWORD)

llm_client = AsyncOpenAI(base_url=LLM_API_ENDPOINT, api_key=TOKEN)

RATE_LIMITER_RPM = AsyncLimiter(max_rate=300, time_period=60)

BASE_PATH = os.getcwd()

with open(f"{BASE_PATH}/prompts.yaml") as f:
    prompts = safe_load(f)




class Worker:
    def __init__(self, queue: asyncio.Queue, results: list[dict], progress_bar: tqdm) -> None:
        self.queue = queue
        self.results = results
        self.progress_bar = progress_bar
        self.llm_client = AsyncOpenAI(base_url=LLM_API_ENDPOINT, api_key=TOKEN)
        self.rate_limiter_rpm = RATE_LIMITER_RPM

    async def _classify(self, purpose: str) -> str:
        async with self.rate_limiter_rpm:
            try:
                result = await self.llm_client.beta.chat.completions.parse(
                    model="large-ft",
                    messages=[
                        {"role": "system", "content": prompts["system_prompt"]},
                        {"role": "user", "content": purpose},
                    ],
                    max_tokens=100,
                    temperature=0.05,
                    response_format=Label
                )
                label = result.choices[0].message.parsed.label
            except Exception as e:
                logger.error(f"Error while classifying {purpose}: {e}")
                label = "NOT_DEFINED"
            return label

    async def process(self, task):
        if task is None:
            return
        idx, purpose, purpose_id = task
        label = await self._classify(purpose)
        self.results[idx]["label"] = label
        self.progress_bar.set_postfix({"label": label})
        self.progress_bar.update(1)

    async def run(self) -> None:
        while True:
            task = await self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            try:
                await self.process(task)
            except Exception as e:
                logger.error(f"Error while processing {task}: {e}")
            finally:
                self.queue.task_done()


async def classify_dataset(input_csv: str, output_csv: str, max_workers: int = 3):
    start_time = time.perf_counter()

    df = pd.read_csv(input_csv)

    task_queue = asyncio.Queue()
    results = [{"purpose_id": df.loc[_, "purpose_id"], "label": None} for _ in range(len(df))]

    with tqdm(total=len(df), desc="Progress", unit="task") as progress_bar:
        workers = [Worker(queue=task_queue, results=results, progress_bar=progress_bar) for _ in range(max_workers)]
        tasks = [asyncio.create_task(worker.run()) for worker in workers]

        for idx, row in df.iterrows():
            await task_queue.put((idx, row["purpose"], row["purpose_id"]))

        await task_queue.join()

        for _ in workers:
            await task_queue.put(None)

        await asyncio.gather(*tasks)

    df["label"] = [result["label"] for result in results]
    df["purpose_id_new"] = [result["purpose_id"] for result in results]
    df.to_csv(output_csv, index=False)

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Data annotation completed in {elapsed_time:.4f} seconds.")

    label_counts = df["label"].value_counts()
    logger.info("Label Statistics:")
    for label, count in label_counts.items():
        logger.info(f"{label}: {count}")


if __name__ == "__main__":
    asyncio.run(classify_dataset("data_annotation.csv", "output.csv"))
