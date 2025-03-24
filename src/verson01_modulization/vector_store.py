import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
import langsmith

import config
from models import EmbeddingModel

class VectorStore:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, base_path: str = None):
        """
        벡터 저장소 초기화
        
        Args:
            base_path: 벡터 저장소 기본 경로
        """
        self.base_path = base_path or config.BASE_PATH
        self.collections = []
        
    def get_available_collections(self) -> List[str]:
        """사용 가능한 모든 컬렉션 목록 반환"""
        return [
            d for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]
    
    def load_collection(self, collection_name: str) -> bool:
        """
        컬렉션 로드
        
        Args:
            collection_name: 로드할 컬렉션 이름
            
        Returns:
            성공 여부 (불리언)
        """
        try:
            # 컬렉션 이름 로깅 (디버깅용)
            print(f"컬렉션 로드 요청 받음: '{collection_name}'")

            # 이미 로드된 컬렉션 확인
            for coll in self.collections:
                if coll["name"] == collection_name:
                    print(f"{collection_name} 컬렉션이 이미 로드되어 있습니다.")
                    return True

            # 기존 알려진 별칭 매핑
            collection_mapping = {
                "db손해보험": "DBSonBo_YakMu20250123",
                "DB손해보험": "DBSonBo_YakMu20250123",
                "db손보": "DBSonBo_YakMu20250123",
                "삼성화재": "Samsung_YakMu2404103NapHae20250113",
                "삼성": "Samsung_YakMu2404103NapHae20250113",
                "DBSonbo_Yakwan20250123": "DBSonBo_YakMu20250123",
            }

            # 로깅을 위한 원래 이름 저장
            original_name = collection_name

            # 매핑 적용
            actual_collection_name = collection_mapping.get(collection_name, collection_name)

            # 원본 이름과 매핑된 이름이 다른 경우 로그 출력
            if original_name != actual_collection_name:
                print(f"자동 변환: '{original_name}' -> '{actual_collection_name}'")

            print(f"컬렉션 로드 시도: '{original_name}' -> '{actual_collection_name}'")

            collection_dir = os.path.join(self.base_path, actual_collection_name)
            
            # 인덱스 파일 찾기
            possible_index_files = ["index.faiss", "faiss.index", "index"]
            index_path = None
            for idx_file in possible_index_files:
                temp_path = os.path.join(collection_dir, idx_file)
                if os.path.exists(temp_path):
                    index_path = temp_path
                    print(f"인덱스 파일을 찾았습니다: {idx_file}")
                    break

            if not index_path:
                raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {collection_dir}")

            # 메타데이터 파일 찾기
            metadata_path = os.path.join(collection_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")

            # 인덱스 로드
            index = faiss.read_index(index_path)

            # 인덱스 타입 확인
            index_type = type(index).__name__
            print(f"로드된 인덱스 타입: {index_type}")
            print(f"인덱스 차원: {index.d}, 벡터 수: {index.ntotal}")

            # L2 인덱스를 내적(코사인 유사도) 인덱스로 변환
            if isinstance(index, faiss.IndexFlatL2) or "L2" in index_type:
                print("유클리드 거리 인덱스를 코사인 유사도 인덱스로 변환합니다...")
                try:
                    # 벡터 추출 시도
                    vectors = index.reconstruct_n(0, index.ntotal)
                    
                    # 벡터 정규화
                    faiss.normalize_L2(vectors)
                    
                    # 새 내적 인덱스 생성
                    new_index = faiss.IndexFlatIP(index.d)
                    new_index.add(vectors)
                    
                    print(f"변환 완료: {index.ntotal}개 벡터가 코사인 유사도 인덱스로 변환됨")
                    index = new_index
                except Exception as e:
                    print(f"인덱스 변환 중 오류: {e}")
                    print("원본 인덱스를 계속 사용합니다.")

            # 메타데이터 파일 로드
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_raw = json.load(f)

            # 메타데이터 형식 확인 및 변환
            if isinstance(metadata_raw, list):
                print(f"메타데이터가 리스트 형식입니다. 딕셔너리로 변환합니다.")
                metadata = {}
                for i, item in enumerate(metadata_raw):
                    metadata[str(i)] = item
                print(f"리스트를 딕셔너리로 변환 완료: {len(metadata)}개 항목")
            else:
                metadata = metadata_raw

            # 컬렉션 추가
            self.collections.append({
                "name": collection_name,
                "index": index,
                "metadata": metadata
            })
            
            print(f"{collection_name} 컬렉션 로드 완료: {len(metadata)}개 벡터")
            return True
            
        except Exception as e:
            print(f"{collection_name} 컬렉션 로드 중 오류: {e}")
            return False
    
    def search(
        self, 
        query_vector: np.ndarray, 
        collection_names: List[str] = None, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행
        
        Args:
            query_vector: 검색할 쿼리 벡터
            collection_names: 검색 대상 컬렉션 이름 목록
            top_k: 각 컬렉션에서 검색할 결과 수
            
        Returns:
            검색 결과 목록
        """
        if not self.collections:
            return [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "로드된 컬렉션이 없습니다."}
            }]

        all_results = []
        use_collections = [
            c for c in self.collections
            if not collection_names or c["name"] in collection_names
        ]
        
        if not use_collections:
            return [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "지정된 컬렉션을 찾을 수 없습니다."}
            }]

        print(f"\n-------- 벡터 검색 시작 --------")
        print(f"대상 컬렉션: {[c['name'] for c in use_collections]}")
        print(f"각 컬렉션당 top_k: {top_k}")

        try:
            # 각 컬렉션에서 항상 top_k개의 문서 검색
            for collection in use_collections:
                try:
                    index = collection["index"]
                    metadata = collection["metadata"]
                    collection_name = collection["name"]

                    print(f"\n검색 중: {collection_name} 컬렉션")
                    print(f"인덱스 차원: {index.d}, 인덱스 타입: {type(index)}")
                    
                    query_dim = query_vector.shape[1]
                    print(f"쿼리 임베딩 차원: {query_dim}")

                    # 차원 불일치 처리
                    if query_dim != index.d:
                        print(f"차원 불일치: 쿼리={query_dim}, 인덱스={index.d}")
                        if query_dim < index.d:
                            # 패딩: 부족한 차원을 0으로 채움
                            padded = np.zeros((1, index.d), dtype=np.float32)
                            padded[0, :query_dim] = query_vector[0, :]
                            query_vector = padded
                            print(f"쿼리 벡터를 {query_dim}에서 {index.d}로 패딩했습니다.")
                        else:
                            # 자름: 여분의 차원을 제거
                            query_vector = query_vector[0, :index.d].reshape(1, -1)
                            print(f"쿼리 벡터를 {query_dim}에서 {index.d}로 잘랐습니다.")

                    # L2 정규화 다시 적용
                    faiss.normalize_L2(query_vector)

                    # 검색 전 벡터 상태 로깅
                    query_norm = np.linalg.norm(query_vector)
                    if abs(query_norm - 1.0) > 1e-5:
                        print(f"경고: 쿼리 벡터 노름이 1이 아닙니다: {query_norm}")
                        # 강제로 정규화
                        query_vector = query_vector / query_norm

                    # 검색 수행
                    distances, indices = index.search(query_vector, top_k)

                    # 내적 값이 1보다 크면 조정
                    if np.any(distances > 1.01):  # 약간의 오차 허용
                        print(f"경고: 내적 값이 1보다 큽니다. 최댓값: {np.max(distances)}")
                        distances = np.minimum(distances, 1.0)

                    # 유사도 점수 변환 (내적 값은 -1~1 범위, 높을수록 유사)
                    # 결과를 0~1 범위로 정규화
                    normalized_scores = (distances + 1) / 2

                    collection_results = []
                    for i, (idx, score) in enumerate(zip(indices[0], normalized_scores[0])):
                        if idx != -1:  # -1은 결과가 없음을 의미
                            # 메타데이터에서 해당 인덱스의 정보 가져오기
                            doc_id = str(idx)
                            
                            # 메타데이터 키가 존재하는지 확인
                            if doc_id in metadata:
                                doc_metadata = metadata[doc_id]
                                collection_results.append({
                                    "collection": collection_name,
                                    "id": doc_id,
                                    "score": float(score),  # 0~1 사이 값, 높을수록 유사
                                    "metadata": doc_metadata
                                })
                            else:
                                # 다른 형식의 키 시도
                                int_id = str(int(idx))
                                if int_id in metadata:
                                    doc_metadata = metadata[int_id]
                                    collection_results.append({
                                        "collection": collection_name,
                                        "id": int_id,
                                        "score": float(score),
                                        "metadata": doc_metadata
                                    })
                                else:
                                    # 모든 방법 실패 시 기본 메타데이터 생성
                                    collection_results.append({
                                        "collection": collection_name,
                                        "id": doc_id,
                                        "score": float(score),
                                        "metadata": {
                                            "text": f"인덱스 {idx}의 메타데이터를 찾을 수 없습니다."
                                        }
                                    })

                    # all_results에 collection_results 추가
                    all_results.extend(collection_results)
                    print(f"{collection_name} 컬렉션에서 {len(collection_results)}개 결과 찾음")
                    
                except Exception as e:
                    print(f"컬렉션 '{collection_name}' 검색 중 오류: {e}")
                    import traceback
                    print(f"상세 오류 정보: {traceback.format_exc()}")
                    continue

            # 결과를 점수에 따라 정렬 (높은 점수가 먼저 오도록)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            print(f"\n총 {len(all_results)}개 청크 검색됨")
            print(f"-------- 벡터 검색 완료 --------\n")
            
            return all_results if all_results else [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "검색 결과를 찾을 수 없습니다."}
            }]
            
        except Exception as e:
            print(f"벡터 검색 중 오류: {e}")
            if config.LANGCHAIN_API_KEY:
                with langsmith.trace(
                    name="vector_search_error",
                    project_name=config.LANGCHAIN_PROJECT,
                    tags=["insupanda", "error"],
                    metadata={"error": str(e)}
                ) as error_run:
                    if error_run:
                        error_run.add_metadata({"error_message": f"검색 오류: {str(e)}"})
            
            return [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": f"검색 중 오류 발생: {str(e)}"}
            }]
    
    def create_index(self, embeddings: np.ndarray, dimension: int = 1024) -> faiss.Index:
        """
        임베딩 배열로부터 FAISS 인덱스를 생성
        
        Args:
            embeddings: 벡터 임베딩 배열
            dimension: 벡터 차원
            
        Returns:
            FAISS 인덱스
        """
        try:
            print(f"인덱스 생성 시작: {len(embeddings)}개 벡터, 차원={dimension}")

            # 모든 벡터가 L2 정규화되어 있는지 확인
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # 정규화 수행
            faiss.normalize_L2(embeddings_array)

            # 내적(코사인 유사도) 기반 인덱스 생성
            index = faiss.IndexFlatIP(dimension)

            # 인덱스에 벡터 추가
            index.add(embeddings_array)

            print(f"인덱스 생성 완료: {index.ntotal}개 벡터")
            return index
        except Exception as e:
            print(f"인덱스 생성 중 오류: {e}")
            raise e 