import asyncio
import time
import uuid
import torch
from loguru import logger

from loans_classification.pipeline import ClassificationPipeline
from loans_classification.database import DataBase
from loans_classification.models import ClassificationTask, ClassificationTaskBatch, LabelConfidencePair, \
    ResponseModelBatch
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
from loans_classification.config import LLM_API_ENDPOINT, LLM_API_AUTH_USERNAME, LLM_API_AUTH_PASSWORD, IS_PREVIEW, \
    IS_TEST, MOCK_ANSWER, MAX_RATE, TIME_PERIOD
from loans_classification.data_annotation.auth import authorize_and_get_token_from_gateway

if IS_TEST or IS_PREVIEW:
    LLM_CLIENT = None
else:
    TOKEN = authorize_and_get_token_from_gateway(username=LLM_API_AUTH_USERNAME, password=LLM_API_AUTH_PASSWORD)
    LLM_CLIENT = AsyncOpenAI(base_url=LLM_API_ENDPOINT, api_key=TOKEN)

RATE_LIMITER_RPM = AsyncLimiter(max_rate=MAX_RATE, time_period=TIME_PERIOD)


class Worker:
    def __init__(self, queue: asyncio.Queue, pipeline: ClassificationPipeline, database: DataBase) -> None:
        self.uuid: str = uuid.uuid4().hex
        self.queue = queue
        self._running = True
        self.pipeline = pipeline
        self.database = database
        self.mapping = {
            ClassificationTask: self.process,
            ClassificationTaskBatch: self.process_list,
        }
        self.rate_limiter_rpm = RATE_LIMITER_RPM
        self.llm_client = LLM_CLIENT

    async def _make_embeddings(self, purposes: list[str]) -> list:
        async with self.rate_limiter_rpm:
            try:
                result = await self.llm_client.embeddings.create(
                    model="m3",
                    input=purposes
                )
                embeddings = [data.embedding for data in result.data]
            except Exception as e:
                logger.error(f"Error while making embeddings for batch: {e}")
                raise RuntimeError("Failed to generate embeddings")
            return embeddings

    @staticmethod
    def _safe_set_future_result(future: asyncio.Future, result):
        if not future.cancelled() and not future.done():
            future.set_result(result)

    @staticmethod
    def _safe_set_future_exception(future: asyncio.Future, exception: Exception):
        if not future.cancelled() and not future.done():
            future.set_exception(exception)

    async def process(self, task: ClassificationTask) -> None:
        if IS_PREVIEW or IS_TEST:
            self._safe_set_future_result(task.future, MOCK_ANSWER)
            return

        start_time = time.time()
        try:
            if not task.text:
                self._safe_set_future_result(task.future, [])
                log_data = {
                    "text": task.text,
                    "predicted_class": "other",
                    "proba": None,
                    "confident": True,
                    "payment_ref": task.payment_ref,
                    "duration": None,
                    "status": "OK"
                }
                if not IS_PREVIEW:
                    await self.database.add_new_log(log_data)
            embedding = await self._make_embeddings([task.text])
            embedding_tensor = await asyncio.to_thread(lambda: torch.tensor(embedding))
            result = await asyncio.to_thread(lambda: self.pipeline.classify(embedding_tensor, [task.text])[0])
            duration = time.time() - start_time

            log_data = {
                "text": task.text,
                "predicted_class": result["label"],
                "proba": result["proba"],
                "confident": result["confident"],
                "payment_ref": task.payment_ref,
                "duration": duration,
                "status": "OK"
            }
            if not IS_PREVIEW:
                await self.database.add_new_log(log_data)

            if result["label"] == "other":
                self._safe_set_future_result(task.future, [])
            else:
                response_data = [{"label": result["label"], "confident": bool(result["confident"])}]
                self._safe_set_future_result(task.future, response_data)
        except Exception as e:
            duration = time.time() - start_time
            error_data = {
                "text": task.text,
                "predicted_class": None,
                "proba": None,
                "confident": None,
                "payment_ref": task.payment_ref,
                "duration": duration,
                "status": str(e)
            }
            if not IS_PREVIEW:
                await self.database.add_new_log(error_data)

            logger.error(f"Worker {self.uuid} failed with error: {e}")
            self._safe_set_future_exception(task.future, e)

    async def process_list(self, task: ClassificationTaskBatch) -> None:
        if IS_PREVIEW or IS_TEST:
            self._safe_set_future_result(task.future,
            [ResponseModelBatch(
                payment_ref=task.data[idx].payment_ref,
                classification=[LabelConfidencePair(
                    label=MOCK_ANSWER[0]["label"],
                    confident=bool(MOCK_ANSWER[0]["confident"])
                )]
            ) for idx in range(len(task.data))])
            return

        start_time = time.time()
        try:
            processed_data = [
                (row, row.text) for row in task.data
            ]
            results = []
            for row, processed_text in processed_data:
                if not processed_text:
                    results.append({
                        "label": "other",
                        "proba": None,
                        "confident": True
                    })
                else:
                    results.append(None)

            non_empty_texts = [text for _, text in processed_data if text]
            non_empty_indices = [i for i, (_, text) in enumerate(processed_data) if text]

            if non_empty_texts:
                embeddings = await self._make_embeddings(non_empty_texts)
                embedding_tensor = await asyncio.to_thread(lambda: torch.tensor(embeddings))
                classified_results = await asyncio.to_thread(
                    lambda: self.pipeline.classify(embedding_tensor, non_empty_texts))

                for idx, classification in zip(non_empty_indices, classified_results):
                    results[idx] = classification

            duration = time.time() - start_time

            response_data = []
            log_data = []
            for idx, result in enumerate(results):
                log_data.append({
                    "text": task.data[idx].text,
                    "predicted_class": result["label"],
                    "proba": result["proba"],
                    "confident": result["confident"],
                    "payment_ref": task.data[idx].payment_ref,
                    "duration": duration,
                    "status": "OK"
                })
                classification_result = []
                if result["label"] != "other":
                    classification_result.append(LabelConfidencePair(
                        label=result["label"],
                        confident=bool(result["confident"])
                    ))
                response_data.append(ResponseModelBatch(
                    payment_ref=task.data[idx].payment_ref,
                    classification=classification_result
                ))
            await self.database.add_new_log(log_data)
            self._safe_set_future_result(task.future, response_data)
        except Exception as e:
            duration = time.time() - start_time
            error_data = [{
                "text": row.text,
                "predicted_class": None,
                "proba": None,
                "confident": None,
                "payment_ref": row.payment_ref,
                "duration": duration,
                "status": str(e)
            } for row in task.data]

            if not IS_PREVIEW:
                await self.database.add_new_log(error_data)

            logger.error(f"Worker {self.uuid} failed with error: {e}")
            self._safe_set_future_exception(task.future, e)
    async def run(self) -> None:
        logger.info(f"Worker {self.uuid} started")

        while self._running:
            task = await self.queue.get()
            if task.future.cancelled():
                logger.info(f"Worker {self.uuid}: Task was cancelled before execution — skipping.")
                self.queue.task_done()
                continue

            handler = self.mapping.get(type(task))
            if not handler:
                logger.error(f"Worker {self.uuid}: Unknown task type {type(task)}")
                self.queue.task_done()
                continue
            try:
                await handler(task)
            except Exception as e:
                logger.error(f"Worker {self.uuid} failed with error: {e} while processing task {task}")
            finally:
                self.queue.task_done()
            logger.info(f"Worker {self.uuid} finished")

    async def stop(self) -> None:
        self._running = False
