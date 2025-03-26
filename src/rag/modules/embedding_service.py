"""
임베딩 생성 및 관리를 담당하는 모듈
"""
import numpy as np
from typing import List, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings


class EmbeddingService:
    """텍스트 임베딩 생성 및 관리를 위한 클래스"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        EmbeddingService 초기화
        
        Args:
            model: 임베딩 모델명
        """
        self.model = model
        self.embeddings = OpenAIEmbeddings(model=model)
    
    def create_embedding(self, text: str) -> List[float]:
        """
        텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            생성된 임베딩 벡터
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 임베딩 반환
            return [0.0] * 1536  # OpenAI 임베딩 차원은 1536
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩 일괄 생성
        
        Args:
            texts: 임베딩할 텍스트 목록
            
        Returns:
            생성된 임베딩 벡터 목록
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"일괄 임베딩 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 임베딩 배열 반환
            return [[0.0] * 1536 for _ in range(len(texts))]
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        임베딩 벡터 정규화
        
        Args:
            embedding: 정규화할 임베딩 벡터
            
        Returns:
            정규화된 임베딩 벡터
        """
        array = np.array(embedding)
        norm = np.linalg.norm(array)
        if norm == 0:
            return embedding
        return (array / norm).tolist()
