import json
import os
import re
from types import MappingProxyType

# Database
USE_DB = json.loads(os.getenv("USE_DB", "false"))
IS_TEST = json.loads(os.getenv("IS_TEST", "false"))
IS_PREVIEW = json.loads(os.getenv("IS_PREVIEW", "false"))
DATABASE_MONITORING_TABLE_NAME = os.getenv("DATABASE_MONITORING_TABLE_NAME", "")
DATABASE_NEW_MONITORING_TABLE_NAME = os.getenv("DATABASE_NEW_MONITORING_TABLE_NAME", "")
DATABASE_SCHEMA_NAME = os.getenv("DATABASE_SCHEMA_NAME", "app")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}@db.cibaa.raiffeisen.ru:5432/{os.getenv('DATABASE_NAME')}"
DATABASE_URL_TEST = f"postgresql+asyncpg://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}@s-msk-t-cibaa-pg-db2.raiffeisen.ru:5432/{os.getenv('DATABASE_NAME')}"

# Workers
NUM_WORKERS = 3

# PyTorch
MODEL_PATHS = {
    'multiclass': 'loans_classification/pytorch_models/best_model_state_dict.pth'
}

_class_thresholds = {'ads': 0.5097097097097097,
                     'loan': 0.2882882882882883,
                     'other': -1,
                     'other_employee_payments': 0.8318318318318318,
                     'package': 0.6116116116116116,
                     'rent': 0.7447447447447447,
                     'salary': 0.16616616616616617,
                     'tax': 0.36736736736736736,
                     'transportation costs': 0.4804804804804805}

CLASS_THRESHOLDS = MappingProxyType(_class_thresholds)

# LLM
LLM_API_ENDPOINT = ""
LLM_API_AUTH_USERNAME = os.getenv("LLM_API_AUTH_USERNAME", "")
LLM_API_AUTH_PASSWORD = os.getenv("LLM_API_AUTH_PASSWORD", "")
MAX_RATE = 300
TIME_PERIOD = 60

# Leasing
LEASING_REGEX = re.compile(
    r"""
    \b(
        лизинг[а-яё]{1,10}         |
        лизенг[а-яё]{1,10}         |
        лизингов[а-яё]{1,10}       |
        лизинг                     |
        лизенг                     |
        лизингов                   |
        фин\.?\s*аренда            |
        финансов\w*\s+аренда       |
        leasing\w*                 |
        lease\w*                   
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)


def is_leasing(text: str) -> bool:
    match = LEASING_REGEX.search(text)
    return bool(match)


# MOCK
MOCK_ANSWER = [{"label": "tax", "confident": True}]
