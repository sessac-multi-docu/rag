"""
컬렉션 로드 및 관리를 담당하는 모듈
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from .insurance_mappings import InsuranceMappings


class CollectionManager:
    """벡터 데이터베이스 컬렉션을 관리하는 클래스"""

    def __init__(self, base_path: str):
        """
        CollectionManager 초기화
        
        Args:
            base_path: 컬렉션 기본 경로
        """
        self.base_path = base_path
        self.collections = []  # 로드된 컬렉션 목록
        self.insurance_mappings = InsuranceMappings()
    
    def get_available_collections(self) -> List[str]:
        """
        사용 가능한 모든 컬렉션 목록 반환
        
        Returns:
            사용 가능한 컬렉션 이름 목록
        """
        available_collections = [
            d for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]
        return available_collections
    
    def is_collection_loaded(self, collection_name: str) -> bool:
        """
        컬렉션이 이미 로드되었는지 확인
        
        Args:
            collection_name: 확인할 컬렉션 이름
            
        Returns:
            컬렉션 로드 여부
        """
        for coll in self.collections:
            if coll["name"] == collection_name:
                return True
        return False
    
    def load_collection(self, collection_name: str) -> bool:
        """
        벡터 데이터베이스 컬렉션 로드
        
        Args:
            collection_name: 로드할 컬렉션 이름
            
        Returns:
            로드 성공 여부
        """
        # 컬렉션 이름 로깅 (디버깅용)
        print(f"컬렉션 로드 요청 받음: '{collection_name}'")
        
        # 이미 로드된 컬렉션 확인
        if self.is_collection_loaded(collection_name):
            print(f"{collection_name} 컬렉션이 이미 로드되어 있습니다.")
            return True
        
        try:
            # FAISS 인덱스 파일 경로
            index_file = os.path.join(self.base_path, collection_name, "faiss.index")
            
            # 메타데이터 파일 경로
            metadata_file = os.path.join(self.base_path, collection_name, "metadata.json")
            
            # FAISS 인덱스 로드
            if not os.path.exists(index_file):
                print(f"오류: {index_file} 파일이 존재하지 않습니다.")
                return False
            
            index = faiss.read_index(index_file)
            
            # 메타데이터 로드
            if not os.path.exists(metadata_file):
                print(f"오류: {metadata_file} 파일이 존재하지 않습니다.")
                return False
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 컬렉션 정보 저장
            collection_info = {
                "name": collection_name,
                "index": index,
                "metadata": metadata
            }
            
            self.collections.append(collection_info)
            print(f"{collection_name} 컬렉션 로드 완료")
            return True
            
        except Exception as e:
            print(f"{collection_name} 컬렉션 로드 중 오류 발생: {str(e)}")
            return False
    
    def find_matching_collections(self, query_text: str, available_collections: List[str]) -> List[str]:
        """
        쿼리 텍스트와 관련된 컬렉션 목록 찾기
        
        Args:
            query_text: 검색 쿼리 텍스트
            available_collections: 사용 가능한 컬렉션 목록
            
        Returns:
            검색 쿼리와 관련된 컬렉션 목록
        """
        # 쿼리에 언급된 보험사 찾기
        mentioned_companies = self.insurance_mappings.detect_companies_in_question(query_text)
        
        # 비교 질문인지 확인 (여러 보험사 비교)
        is_comparison = self.insurance_mappings.is_comparison_question(query_text)
        
        matched_collections = []
        
        if mentioned_companies:
            # 언급된 보험사와 관련된 컬렉션만 포함
            for company in mentioned_companies:
                for keyword in self.insurance_mappings.company_keywords[company]:
                    collection = self.insurance_mappings.get_collection_name(keyword)
                    if collection and collection in available_collections:
                        matched_collections.append(collection)
                        break
            
            # 중복 제거
            matched_collections = list(set(matched_collections))
            
        # 보험사가 언급되지 않았거나 모든 보험사 비교가 필요할 경우
        if not matched_collections or ("모든" in query_text and is_comparison):
            return available_collections
        
        return matched_collections
    
    def get_all_loaded_collections(self) -> List[Dict[str, Any]]:
        """
        현재 로드된 모든 컬렉션 반환
        
        Returns:
            로드된 컬렉션 목록
        """
        return self.collections
    
    def get_collection_by_name(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        컬렉션 이름으로 컬렉션 정보 조회
        
        Args:
            collection_name: 조회할 컬렉션 이름
            
        Returns:
            컬렉션 정보 (없으면 None)
        """
        for coll in self.collections:
            if coll["name"] == collection_name:
                return coll
        return None
    
    def clear_collections(self) -> None:
        """모든 로드된 컬렉션 초기화"""
        self.collections = []
        print("모든 컬렉션이 초기화되었습니다.")
