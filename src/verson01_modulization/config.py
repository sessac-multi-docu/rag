import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키 관련 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith 설정
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "insupanda")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# LangSmith 트래킹 활성화
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print("LangSmith 트래킹이 활성화되었습니다.")
else:
    print("LangSmith API 키가 설정되지 않았습니다. 트래킹은 비활성화됩니다.")

# 경로 설정
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db")

# 모델 설정
DEFAULT_EMBEDDING_MODEL = "embedding-query"
DEFAULT_LLM_MODEL = "gpt-4o-mini" 