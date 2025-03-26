import os
import json
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


class CollectionLoader:
    def __init__(self, base_path: str, embeddings: Embeddings):
        self.base_path = base_path
        self.embeddings = embeddings
        self.collections = []

    def load_collection(self, collection_name: str) -> bool:
        try:
            print(f"컬렉션 로드 요청 받음: '{collection_name}'")

            for coll in self.collections:
                if coll["name"] == collection_name:
                    print(f"{collection_name} 컬렉션이 이미 로드되어 있습니다.")
                    return True

            collection_mapping = {
                "db손해보험": "DBSonBo_YakMu20250123",
                "DB손해보험": "DBSonBo_YakMu20250123",
                "db손보": "DBSonBo_YakMu20250123",
                "삼성화재": "Samsung_YakMu2404103NapHae20250113",
                "삼성": "Samsung_YakMu2404103NapHae20250113",
                "DBSonbo_Yakwan20250123": "DBSonBo_YakMu20250123",
            }

            original_name = collection_name
            actual_collection_name = collection_mapping.get(
                collection_name, collection_name
            )

            if original_name != actual_collection_name:
                print(f"자동 변환: '{original_name}' -> '{actual_collection_name}'")

            collection_dir = os.path.join(self.base_path, actual_collection_name)
            if not os.path.exists(collection_dir):
                print(f"경로가 존재하지 않음: {collection_dir}")
                print("사용 가능한 디렉토리:")
                for d in os.listdir(self.base_path):
                    print(f"  - {d}")
                return False

            print(f"LangChain을 통해 컬렉션 로드 시도: '{collection_dir}'")

            # LangChain의 FAISS.load_local 사용
            vectordb = FAISS.load_local(
                folder_path=collection_dir,
                embeddings=self.embeddings,
                index_name="faiss",  # 보통 faiss.index 또는 index.faiss 중 하나
            )

            # 메타데이터 개수 확인
            vector_count = vectordb.index.ntotal
            print(f"컬렉션 로드 완료: {vector_count}개 벡터")

            self.collections.append(
                {
                    "name": collection_name,
                    "vectordb": vectordb,
                }
            )

            return True

        except Exception as e:
            print(f"{collection_name} 컬렉션 로드 중 오류: {e}")
            return False
