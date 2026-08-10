import os
from dotenv import load_dotenv

load_dotenv(override=True)

# google_api_key = os.getenv("GOOGLE_API_KEY")

# # Cloud model used only after the CV has been redacted.
# MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
# MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# MODEL_API_KEY = os.getenv("MODEL_API_KEY", google_api_key)
# MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.0))

# Local model used to inspect the raw CV for PII.
MODEL_NAME = os.getenv("PII_MODEL_NAME", "llama3.2:latest")
MODEL_BASE_URL = os.getenv("PII_MODEL_BASE_URL", "http://localhost:11434/v1")
MODEL_API_KEY = os.getenv("PII_MODEL_API_KEY", "ollama")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.0))

# Local model used to inspect the raw CV for PII.
PII_MODEL_NAME = os.getenv("PII_MODEL_NAME", "llama3.2:latest")
PII_MODEL_BASE_URL = os.getenv("PII_MODEL_BASE_URL", "http://localhost:11434/v1")
PII_MODEL_API_KEY = os.getenv("PII_MODEL_API_KEY", "ollama")

SKILLS_MATCH_WEIGHT = float(os.getenv("SKILLS_MATCH_WEIGHT", 0.45))
SKILL_TENURE_WEIGHT = float(os.getenv("SKILL_TENURE_WEIGHT", 0.35))
WORK_EXPERIENCE_WEIGHT = float(os.getenv("WORK_EXPERIENCE_WEIGHT", 0.20))