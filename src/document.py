import os
import json
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.document_loaders import JSONLoader

# 기본 설정
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db")
COLLECTION_NAME = "Samsung_YakMu2404103NapHae20250113"  # 예시 이름
collection_dir = os.path.join(VECTOR_DB_PATH, COLLECTION_NAME)

# 벡터 인덱스와 메타데이터 파일 경로 설정
index_file_candidates = ["index.faiss", "faiss.index", "index"]
index_path = None
for fname in index_file_candidates:
    path = os.path.join(collection_dir, fname)
    if os.path.exists(path):
        index_path = path
        break

metadata_path = os.path.join(collection_dir, "metadata.json")
# 메타데이터 로딩
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata_raw = json.load(f)

# 리스트 형식인 경우 딕셔너리로 변환
if isinstance(metadata_raw, list):
    metadata = {str(i): chunk for i, chunk in enumerate(metadata_raw)}
else:
    metadata = metadata_raw

# LangChain용 FAISS 인스턴스 생성 (기존 인덱스를 불러와서 래핑)
# embedding = OpenAIEmbeddings()
# vectorstore = FAISS.load_local(folder_path=collection_dir, embeddings=embedding)

print("FAISS 벡터 DB 로드 완료:", COLLECTION_NAME)
print("총 문서 수:", len(metadata))

if __name__ == "__main__":
    file_path = "/Users/hyerim/sessac/rag/src/vector_db"

    available_collections = os.listdir(file_path)
