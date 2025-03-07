import json
import os

from dotenv import load_dotenv

load_dotenv()


def load_settings(user_path):
    settings_file = os.path.join(user_path, "settings.json")
    default_settings = {
        "source": "OpenAI",
        "chat_model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "embedding_model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME"),
        "hf_token": os.getenv("HUGGINGFACE_API_TOKEN"),
        "chunk_size": 512,
        "chunk_overlap": 25,
        "k": 4,
        "score_threshold": 0.75,
        "system_prompt": "あなたは優秀な医療アシスタントです。ユーザーの質問に対して、正確で信頼性の高い医療知識を基に分かりやすく回答してください。",
        "data_dir": "./data",
    }

    if not os.path.exists(settings_file):
        save_settings(user_path, default_settings)
        return default_settings

    with open(settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)
        merged_settings = {**default_settings, **settings}
        return merged_settings


def save_settings(user_path, settings):
    settings_file = os.path.join(user_path, "settings.json")
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
