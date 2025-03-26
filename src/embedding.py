from langchain_upstage import UpstageEmbeddings
import numpy as np
import faiss


def get_upstage_embedding(self, text):
    if text in self.cached_embeddings:
        return self.cached_embeddings[text]

    if not self.api_key or len(self.api_key) < 10:
        raise ValueError(f"유효한 Upstage API 키가 없습니다. 현재 키: {self.api_key}")

    try:
        # UpstageEmbeddings 인스턴스 (1회만 생성되도록 클래스 외부에서 관리하는 게 좋음)
        upstage = UpstageEmbeddings(
            api_key=self.api_key, model="solar-embedding-1-large"
        )

        # 텍스트 임베딩 생성
        embedding = upstage.embed_query(text)
        print(f"임베딩 생성 성공, 차원: {len(embedding)}")

        # 벡터 변환 및 정규화
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        original_dim = vector.shape[1]
        norm_before = np.linalg.norm(vector)
        print(f"정규화 전 벡터 노름: {norm_before}")

        faiss.normalize_L2(vector)

        norm_after = np.linalg.norm(vector)
        print(f"정규화 후 벡터 노름: {norm_after}")
        print(
            f"벡터 통계: 최소={np.min(vector):.4f}, 최대={np.max(vector):.4f}, 평균={np.mean(vector):.4f}"
        )
        print(f"내적 기반 검색을 위해 준비된 임베딩 형태: {vector.shape}")

        # 결과 캐시
        self.cached_embeddings[text] = vector
        return vector

    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        raise ValueError(
            "임베딩 생성에 실패했습니다. API 키와 네트워크 상태를 확인하세요."
        )
