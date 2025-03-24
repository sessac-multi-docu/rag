import numpy as np
import faiss
import langsmith
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
import os

from openai import OpenAI
from langsmith.wrappers import wrap_openai

import config

class EmbeddingModel(ABC):
    """임베딩 모델의 추상 클래스"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        pass
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """벡터 정규화"""
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # 정규화 이전 벡터 노름 계산
        norm_before = np.linalg.norm(vector)
        print(f"정규화 전 벡터 노름: {norm_before}")
        
        # L2 정규화 수행
        faiss.normalize_L2(vector)
        
        # 정규화 이후 벡터 노름 확인
        norm_after = np.linalg.norm(vector)
        print(f"정규화 후 벡터 노름: {norm_after}")
        
        return vector


class UpstageEmbeddingModel(EmbeddingModel):
    """Upstage API를 사용한 임베딩 모델"""
    
    def __init__(self, api_key: str = None, model_name: str = "embedding-query"):
        self.api_key = api_key or config.UPSTAGE_API_KEY
        self.model_name = model_name
        self.cached_embeddings = {}
        
        if not self.api_key or len(self.api_key) < 10:
            raise ValueError(f"유효한 Upstage API 키가 없습니다. 현재 키: {self.api_key}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Upstage API를 통해 임베딩 벡터 생성"""
        if text in self.cached_embeddings:
            return self.cached_embeddings[text]
            
        try:
            # OpenAI 클라이언트 생성
            client = OpenAI(
                api_key=self.api_key, 
                base_url="https://api.upstage.ai/v1/solar"
            )
            
            # LangSmith 트래킹이 활성화된 경우 클라이언트 래핑
            if os.getenv("LANGCHAIN_API_KEY"):
                client = wrap_openai(client)  # 인스턴스를 래핑
                
            # 임베딩 생성 요청과 트래킹
            with langsmith.trace(
                name="upstage_embedding",
                project_name=config.LANGCHAIN_PROJECT,
                tags=["insupanda", "embedding"],
                metadata={"text_length": len(text)},
            ) as run:
                response = client.embeddings.create(
                    input=text, 
                    model=self.model_name
                )
                embedding = response.data[0].embedding
                
                print(f"임베딩 생성 성공, 차원: {len(embedding)}")
                
                vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
                # 원본 벡터 차원 저장
                original_dim = vector.shape[1]
                
                # 정규화 및 벡터 통계치 확인
                vector = self.normalize_vector(vector)
                print(f"벡터 통계: 최소값={np.min(vector)}, 최대값={np.max(vector)}, 평균={np.mean(vector)}")
                print(f"내적 기반 검색을 위해 준비된 임베딩 형태: {vector.shape}")
                
                # 결과 캐싱 및 메타데이터 추가
                self.cached_embeddings[text] = vector
                if run:
                    run.add_metadata({
                        "embedding_dimension": original_dim,
                        "norm_before": float(np.linalg.norm(vector)),
                        "norm_after": float(np.linalg.norm(vector)),
                        "success": True,
                    })
                
                return vector
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            if os.getenv("LANGCHAIN_API_KEY"):
                with langsmith.trace(
                    name="embedding_error",
                    project_name=config.LANGCHAIN_PROJECT,
                    tags=["insupanda", "error", "embedding"],
                    metadata={"error": str(e)},
                ) as error_run:
                    if error_run:
                        error_run.add_metadata(
                            {"error_message": f"임베딩 생성 오류: {str(e)}"}
                        )
            raise ValueError("임베딩 생성에 실패했습니다. API 키와 네트워크 상태를 확인하세요.")


class MockEmbeddingModel(EmbeddingModel):
    """테스트 목적의 모의 임베딩 모델"""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        
    def get_embedding(self, text: str) -> np.ndarray:
        """랜덤 임베딩 벡터 생성 (테스트용)"""
        # 해시 함수를 사용하여 동일한 문자열에 대해 동일한 임베딩 생성
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        vector = np.random.randn(1, self.dimension).astype(np.float32)
        return self.normalize_vector(vector)


def get_embedding_model(model_type: str = "upstage", **kwargs) -> EmbeddingModel:
    """팩토리 메서드: 지정된 유형의 임베딩 모델 인스턴스 생성"""
    if model_type.lower() == "upstage":
        return UpstageEmbeddingModel(**kwargs)
    elif model_type.lower() == "mock":
        return MockEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"지원되지 않는 임베딩 모델 유형: {model_type}") 