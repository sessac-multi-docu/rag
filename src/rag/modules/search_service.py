"""
벡터 검색 기능을 담당하는 모듈
"""
import numpy as np
from typing import List, Dict, Any, Union
from .embedding_service import EmbeddingService
from .collection_manager import CollectionManager


class SearchService:
    """벡터 검색을 수행하는 클래스"""
    
    def __init__(self, collection_manager: CollectionManager, embedding_service: EmbeddingService):
        """
        SearchService 초기화
        
        Args:
            collection_manager: 컬렉션 관리자 인스턴스
            embedding_service: 임베딩 서비스 인스턴스
        """
        self.collection_manager = collection_manager
        self.embedding_service = embedding_service
    
    def search(self, query_text: str, collection_names: List[str], top_k: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        검색 쿼리에 대한 유사 문서 검색
        
        Args:
            query_text: 검색할 쿼리 텍스트
            collection_names: 검색할 컬렉션 이름 목록
            top_k: 각 컬렉션별 반환할 최대 결과 수
            
        Returns:
            컬렉션별 검색 결과 딕셔너리
        """
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_service.create_embedding(query_text)
        
        # 결과 저장 딕셔너리
        results = {}
        
        # 각 컬렉션별로 검색 수행
        for collection_name in collection_names:
            collection = self.collection_manager.get_collection_by_name(collection_name)
            if not collection:
                continue
            
            try:
                # FAISS 검색 실행
                index = collection["index"]
                metadata = collection["metadata"]
                
                # 임베딩을 numpy 배열로 변환
                xq = np.array([query_embedding], dtype=np.float32)
                
                # 검색 수행 (거리, 인덱스 반환)
                distances, indices = index.search(xq, top_k)
                
                # 검색 결과 포맷팅
                collection_results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(metadata):  # 유효한 인덱스인지 확인
                        result = {
                            "content": metadata[idx]["content"],
                            "source": metadata[idx]["source"],
                            "score": float(1.0 - distances[0][i]),  # 거리를 유사도 점수로 변환
                        }
                        collection_results.append(result)
                
                # 결과가 있는 경우만 저장
                if collection_results:
                    results[collection_name] = collection_results
                    
            except Exception as e:
                print(f"{collection_name} 컬렉션 검색 중 오류 발생: {str(e)}")
        
        return results
    
    def search_all_collections(self, query_text: str, top_k: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        로드된 모든 컬렉션에서 검색 수행
        
        Args:
            query_text: 검색할 쿼리 텍스트
            top_k: 각 컬렉션별 반환할 최대 결과 수
            
        Returns:
            컬렉션별 검색 결과 딕셔너리
        """
        loaded_collections = self.collection_manager.get_all_loaded_collections()
        collection_names = [coll["name"] for coll in loaded_collections]
        return self.search(query_text, collection_names, top_k)
    
    def format_search_results(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        검색 결과를 사람이 읽기 쉬운 형태로 포맷팅
        
        Args:
            search_results: 검색 결과 딕셔너리
            
        Returns:
            포맷팅된 검색 결과 텍스트
        """
        formatted_text = ""
        
        if not search_results:
            return "검색 결과가 없습니다."
        
        for collection_name, results in search_results.items():
            formatted_text += f"== {collection_name} 컬렉션 검색 결과 ==\n\n"
            
            for i, result in enumerate(results):
                score = result.get("score", 0.0)
                source = result.get("source", "알 수 없는 출처")
                content = result.get("content", "내용 없음")
                
                formatted_text += f"[결과 {i+1}] 유사도: {score:.4f}\n"
                formatted_text += f"출처: {source}\n"
                formatted_text += f"내용: {content}\n\n"
        
        return formatted_text
