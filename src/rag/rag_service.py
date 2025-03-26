"""
RAG 시스템의 메인 서비스 클래스
"""
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .modules.collection_manager import CollectionManager
from .modules.embedding_service import EmbeddingService
from .modules.search_service import SearchService
from .modules.answer_generator import AnswerGenerator
from .modules.insurance_mappings import InsuranceMappings


@dataclass
class SearchQuery:
    """검색 쿼리 데이터 클래스"""
    query_text: str
    collections: List[str] = None
    
    def __post_init__(self):
        if self.collections is None:
            self.collections = []


class RAGService:
    """보험 약관 RAG 시스템의 메인 서비스 클래스"""
    
    def __init__(self, base_path: str = "vector_db", openai_api_key: Optional[str] = None):
        """
        RAGService 초기화
        
        Args:
            base_path: 벡터 데이터베이스 기본 경로
            openai_api_key: OpenAI API 키
        """
        self.base_path = base_path
        
        # 모듈 초기화
        self.insurance_mappings = InsuranceMappings()
        self.collection_manager = CollectionManager(base_path)
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService(self.collection_manager, self.embedding_service)
        self.answer_generator = AnswerGenerator(openai_api_key)
        
        print(f"RAG 서비스 초기화 완료. 벡터 DB 경로: {base_path}")
    
    def find_matching_collections(self, query_text: str, available_collections: List[str]) -> List[str]:
        """
        쿼리 텍스트와 관련된 컬렉션을 찾음
        
        Args:
            query_text: 검색 쿼리 텍스트
            available_collections: 사용 가능한 컬렉션 목록
            
        Returns:
            관련 컬렉션 목록
        """
        return self.collection_manager.find_matching_collections(query_text, available_collections)
    
    def load_collection(self, collection_name: str) -> bool:
        """
        컬렉션 로드
        
        Args:
            collection_name: 로드할 컬렉션 이름
            
        Returns:
            로드 성공 여부
        """
        return self.collection_manager.load_collection(collection_name)
    
    def search(self, query_text: str, collection_names: List[str], top_k: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        쿼리 텍스트로 검색 수행
        
        Args:
            query_text: 검색 쿼리 텍스트
            collection_names: 검색할 컬렉션 이름 목록
            top_k: 각 컬렉션별 반환할 최대 결과 수
            
        Returns:
            컬렉션별 검색 결과 딕셔너리
        """
        return self.search_service.search(query_text, collection_names, top_k)
    
    def generate_answer(self, query_text: str, search_results: Dict[str, List[Dict[str, Any]]], openai_api_key: Optional[str] = None, model: str = "gpt-4-turbo") -> str:
        """
        검색 결과를 기반으로 답변 생성
        
        Args:
            query_text: 사용자 질문
            search_results: 검색 결과 딕셔너리
            openai_api_key: OpenAI API 키 (선택)
            model: 사용할 OpenAI 모델
            
        Returns:
            생성된 답변
        """
        if openai_api_key:
            self.answer_generator.openai_api_key = openai_api_key
        
        return self.answer_generator.generate_answer(query_text, search_results, model)
    
    def process_query(self, query: Union[str, SearchQuery], top_k: int = 2, model: str = "gpt-4-turbo") -> str:
        """
        쿼리 처리 파이프라인 실행 (검색 및 답변 생성)
        
        Args:
            query: 검색 쿼리 문자열 또는 SearchQuery 객체
            top_k: 각 컬렉션별 반환할 최대 결과 수
            model: 사용할 OpenAI 모델
            
        Returns:
            생성된 답변
        """
        # 만약 query가 문자열이면 SearchQuery 객체로 변환
        if isinstance(query, str):
            query = SearchQuery(query_text=query, collections=[])
        
        # 사용 가능한 컬렉션 목록 가져오기
        available_collections = self.collection_manager.get_available_collections()
        print(f"사용 가능한 컬렉션: {available_collections}")
        
        # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾음
        use_collections = (
            query.collections
            if query.collections
            else self.find_matching_collections(query.query_text, available_collections)
        )
        print(f"사용할 컬렉션: {use_collections}")
        
        # 찾은 컬렉션 로드
        for collection_name in use_collections:
            self.load_collection(collection_name)
        
        # 검색 수행
        search_results = self.search(query.query_text, use_collections, top_k)
        
        # 답변 생성
        answer = self.generate_answer(query.query_text, search_results, model=model)
        
        return answer
    
    def clear_collections(self) -> None:
        """로드된 모든 컬렉션 초기화"""
        self.collection_manager.clear_collections()
    
    def get_company_name(self, collection_name: str) -> str:
        """
        컬렉션 이름에 해당하는 보험사 이름 반환
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            보험사 이름
        """
        return self.insurance_mappings.get_company_name(collection_name)
