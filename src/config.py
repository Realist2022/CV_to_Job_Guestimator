import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Engine Settings
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:latest")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "ollama")
MODEL_TEMPERATURE = 0.0

# Three-Pillar Score Weights
SKILLS_MATCH_WEIGHT = 0.45
SKILL_TENURE_WEIGHT = 0.35
WORK_EXPERIENCE_WEIGHT = 0.20