import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from asyncio import Queue

from loans_classification.config import MODEL_PATHS, NUM_WORKERS
from loans_classification.database import DataBase
from loans_classification.model_service import ModelService
from loans_classification.pipeline import ClassificationPipeline
from loans_classification.worker import Worker
from loans_classification.models import ClassificationTask, LabelConfidencePair, ResponseModelBatch, PredictionInput, \
    ClassificationTaskBatch

model_service = ModelService(MODEL_PATHS)
pipeline = ClassificationPipeline(model_service)
database = DataBase()

task_queue: Queue = Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models on startup...")
    model_service.load_all_models()
    workers = [Worker(task_queue, pipeline, database) for _ in range(NUM_WORKERS)]
    worker_tasks = [
        asyncio.create_task(one_worker.run(), name=f"Worker {one_worker.uuid}")
        for one_worker in workers
    ]

    yield

    for one_worker in workers:
        await one_worker.stop()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    logger.info("Unloading models on shutdown...")
    model_service.unload_all_models()


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=list[LabelConfidencePair])
async def predict(input_data: PredictionInput):
    future = asyncio.Future()
    task = ClassificationTask(text=input_data.text, payment_ref=input_data.payment_ref, future=future)
    await task_queue.put(task)
    try:
        result = await asyncio.wait_for(future, timeout=15)
        return result
    except asyncio.TimeoutError:
        task.future.cancel()
        raise HTTPException(status_code=504, detail="Timeout while waiting for inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictByList", response_model=list[ResponseModelBatch])
async def predict_list(input_data: list[PredictionInput]):
    future = asyncio.Future()
    task = ClassificationTaskBatch(data=input_data, future=future)
    await task_queue.put(task)
    try:
        result = await asyncio.wait_for(future, timeout=30)
        return result
    except asyncio.TimeoutError:
        task.future.cancel()
        raise HTTPException(status_code=504, detail="Timeout while waiting for inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, port=8078)
