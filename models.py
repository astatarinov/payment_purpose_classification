import asyncio
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class PredictionInput(BaseModel):
    text: str
    payment_ref: str


class PredictionInputBatch(BaseModel):
    data: list[PredictionInput] = Field(..., description="Список предсказаний с метками и уверенностью")


class LabelConfidencePair(BaseModel):
    label: str = Field(...,
                       description="Название класса, например: 'tax'. Если пусто (''), класс не определён или равен 'other'.")
    confident: bool | None = Field(..., description="Уверенность модели в предсказании для данного класса.")


class ResponseModel(BaseModel):
    results: list[LabelConfidencePair] = Field(..., description="Список предсказаний с метками и уверенностью")


class ResponseModelBatch(BaseModel):
    payment_ref: str = Field(..., description="Ссылка на платеж.")
    classification: list[LabelConfidencePair | None] = Field(
        ...,
        description="Список предсказаний с метками и уверенностью"
    )


class ClassificationTask(BaseModel):
    text: str
    payment_ref: str
    future: asyncio.Future
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassificationTaskBatch(BaseModel):
    data: list[PredictionInput]
    future: asyncio.Future
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Label(BaseModel):
    label: Literal['transportation costs', 'tax', 'other', 'salary', 'rent',
    'office expenses', 'loan', 'ads', 'other_employee_payments', 'package', 'leasing']
