import os
import json
import numpy as np
import faiss

from pydantic import BaseModel
from typing import List, Union

from dotenv import load_dotenv
# LangSmith 트래킹 관련 임포트 수정
from langsmith import Client, traceable
import langsmith
# OpenAI 래핑을 위한 임포트
from langsmith.wrappers import wrap_openai

load_dotenv()

# LangChain 임포트
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# LangSmith 설정
# 이 변수들은 환경 변수에서 가져오거나 직접 설정할 수 있습니다
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "insupanda")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# LangSmith 트래킹 활성화 (환경 변수로도 설정 가능)
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

    # LangSmith 클라이언트 초기화
    # langsmith_client = Client(api_key=LANGCHAIN_API_KEY, api_url=LANGCHAIN_ENDPOINT)
    # print("LangSmith 트래킹이 활성화되었습니다.")

    # OpenAI 래핑 설정 (추후 OpenAI 클라이언트 사용 시)
    try:
        # OpenAI 클래스를 직접 래핑하는 대신 각 인스턴스별로 래핑
        # from openai import OpenAI
        # wrapped_openai = wrap_openai(OpenAI)  # 이 방식은 작동하지 않음
        print("OpenAI 래핑은 get_upstage_embedding 메서드에서 개별적으로 수행됨")
    except ImportError:
        print("OpenAI SDK를 가져올 수 없습니다. OpenAI 래핑은 건너뜁니다.")
else:
    print("LangSmith API 키가 설정되지 않았습니다. 트래킹은 비활성화됩니다.")


class NestedQuery(BaseModel):
    query: str


class SearchQuery(BaseModel):
    query: Union[str, NestedQuery]
    collections: List[str] = None

    class Config:
        # 추가 유효성 검사와 예제 값 설정
        schema_extra = {
            "example": {
                "query": "삼성화재 암보험에 대해 알려줘",
                "collections": ["Samsung_YakMu2404103NapHae20250113"],
            }
        }

    @property
    def query_text(self) -> str:
        """쿼리 텍스트를 추출합니다 (문자열이나 NestedQuery 객체에서)"""
        if isinstance(self.query, str):
            return self.query
        return self.query.query


def get_friendly_name(collection_name: str) -> str:
    return collection_name


class RAGService:
    def __init__(self, upstage_api_key=None):
        self.api_key = upstage_api_key or os.getenv("UPSTAGE_API_KEY")
        self.collections = []
        self.cached_embeddings = {}
        self.base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "src/vector_db"
        )
        self.collection_to_company_mapping = {
            "DBSonBo_YakMu20250123": "DB손해보험",
            "Samsung_YakMu2404103NapHae20250113": "삼성화재",
            "HaNa_YakMuHaGaengPyo20250101": "하나손해보험",
            "HanWha_YakHan20250201": "한화손해보험",
            "Heung_YakMu250220250205": "흥국화재",
            "HyunDai_YakMuSeH1Il2Nap20250213": "현대해상",
            "KB_YakKSeHaeMu250120250214": "KB손해보험",
            "LotteSonBo_YakMuLDeo25011220250101": "롯데손해보험",
            "MGSonBo_YakMuWon2404Se20250101": "MG손해보험",
            "Meritz_YakMu220250113": "메리츠화재",
            "NH_YakMuN5250120250101": "NH농협손해보험"
        }

    def load_collection(self, collection_name):
        try:
            # 컬렉션 이름 로깅 (디버깅용)
            print(f"컬렉션 로드 요청 받음: '{collection_name}'")

            # 이미 로드된 컬렉션 확인
            for coll in self.collections:
                if coll["name"] == collection_name:
                    print(f"{collection_name} 컬렉션이 이미 로드되어 있습니다.")
                    return True

            # 하드코딩된 매핑 대신 동적으로 컬렉션 이름 사용
            # 예외 처리: 알려진 별칭이 있는 경우 실제 디렉토리 이름으로 매핑
            collection_mapping = {
                # DB손해보험 관련 매핑
                "db손해보험": "DBSonBo_YakMu20250123",
                "DB손해보험": "DBSonBo_YakMu20250123",
                "db손보": "DBSonBo_YakMu20250123",
                "디비손해보험": "DBSonBo_YakMu20250123",
                "디비손보": "DBSonBo_YakMu20250123",
                "디비": "DBSonBo_YakMu20250123",
                "DBSonbo_Yakwan20250123": "DBSonBo_YakMu20250123",

                # 삼성화재 관련 매핑
                "삼성화재": "Samsung_YakMu2404103NapHae20250113",
                "삼성": "Samsung_YakMu2404103NapHae20250113",
                "samsung": "Samsung_YakMu2404103NapHae20250113",

                # 하나손해보험 관련 매핑
                "하나손해보험": "HaNa_YakMuHaGaengPyo20250101",
                "하나손보": "HaNa_YakMuHaGaengPyo20250101",
                "하나": "HaNa_YakMuHaGaengPyo20250101",
                "hana": "HaNa_YakMuHaGaengPyo20250101",

                # 한화손해보험 관련 매핑
                "한화손해보험": "HanWha_YakHan20250201",
                "한화손보": "HanWha_YakHan20250201",
                "한화": "HanWha_YakHan20250201",
                "hanwha": "HanWha_YakHan20250201",

                # 흥국화재 관련 매핑
                "흥국화재": "Heung_YakMu250220250205",
                "흥국": "Heung_YakMu250220250205",
                "heung": "Heung_YakMu250220250205",
                

                # 현대해상 관련 매핑
                "현대해상": "HyunDai_YakMuSeH1Il2Nap20250213",
                "현대": "HyunDai_YakMuSeH1Il2Nap20250213",
                "hyundai": "HyunDai_YakMuSeH1Il2Nap20250213",

                # KB손해보험 관련 매핑
                "KB손해보험": "KB_YakKSeHaeMu250120250214",
                "KB손보": "KB_YakKSeHaeMu250120250214",
                "KB": "KB_YakKSeHaeMu250120250214",
                "케이비": "KB_YakKSeHaeMu250120250214",

                # 롯데손해보험 관련 매핑
                "롯데손해보험": "LotteSonBo_YakMuLDeo25011220250101",
                "롯데손보": "LotteSonBo_YakMuLDeo25011220250101",
                "롯데": "LotteSonBo_YakMuLDeo25011220250101",
                "lotte": "LotteSonBo_YakMuLDeo25011220250101",

                # MG손해보험 관련 매핑
                "MG손해보험": "MGSonBo_YakMuWon2404Se20250101",
                "MG손보": "MGSonBo_YakMuWon2404Se20250101",
                "MG": "MGSonBo_YakMuWon2404Se20250101",
                "엠지": "MGSonBo_YakMuWon2404Se20250101",

                # 메리츠화재 관련 매핑
                "메리츠화재": "Meritz_YakMu220250113",
                "메리츠": "Meritz_YakMu220250113",
                "meritz": "Meritz_YakMu220250113",

                # NH농협손해보험 관련 매핑
                "NH농협손해보험": "NH_YakMuN5250120250101",
                "NH손해보험": "NH_YakMuN5250120250101",
                "농협손해보험": "NH_YakMuN5250120250101",
                "NH손보": "NH_YakMuN5250120250101",
                "농협손보": "NH_YakMuN5250120250101",
                "NH": "NH_YakMuN5250120250101",
                "농협": "NH_YakMuN5250120250101",
            }

            # 로깅을 위한 원래 이름 저장
            original_name = collection_name

            # 매핑 적용
            actual_collection_name = collection_mapping.get(
                collection_name, collection_name
            )

            # 원본 이름과 매핑된 이름이 다른 경우 로그 출력
            if original_name != actual_collection_name:
                print(f"자동 변환: '{original_name}' -> '{actual_collection_name}'")

            # 추가 로깅으로 클라이언트 요청 정보 기록
            print(
                f"컬렉션 요청 정보: 원본={original_name}, 매핑 후={actual_collection_name}"
            )

            # 만약 존재하지 않는 경로일 경우 추가 처리
            if not os.path.exists(os.path.join(self.base_path, actual_collection_name)):
                print(
                    f"경로가 존재하지 않음: {os.path.join(self.base_path, actual_collection_name)}"
                )

                # 실제 존재하는 모든 컬렉션 디렉토리 확인
                print(f"사용 가능한 컬렉션 디렉토리:")
                for dir_name in os.listdir(self.base_path):
                    print(f"  - {dir_name}")

                # sonbo/SONBO/Sonbo 대소문자 문제 처리
                if (
                    "sonbo" in actual_collection_name.lower()
                    or "손보" in actual_collection_name
                ):
                    print("DB손해보험 관련 컬렉션 찾는 중...")
                    # 실제 존재하는 디렉토리 찾기
                    for dir_name in os.listdir(self.base_path):
                        if "sonbo" in dir_name.lower() or "SonBo" in dir_name:
                            actual_collection_name = dir_name
                            print(f"DB손해보험 컬렉션 찾음: {dir_name}")
                            break

            print(f"컬렉션 로드 시도: '{original_name}' -> '{actual_collection_name}'")

            collection_dir = os.path.join(self.base_path, actual_collection_name)
            possible_index_files = ["index.faiss", "faiss.index", "index"]
            index_path = None
            for idx_file in possible_index_files:
                temp_path = os.path.join(collection_dir, idx_file)
                if os.path.exists(temp_path):
                    index_path = temp_path
                    print(f"인덱스 파일을 찾았습니다: {idx_file}")
                    break

            if not index_path:
                raise FileNotFoundError(
                    f"인덱스 파일을 찾을 수 없습니다: {collection_dir}"
                )

            metadata_path = os.path.join(collection_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}"
                )

            index = faiss.read_index(index_path)

            # 인덱스 타입 확인
            index_type = type(index).__name__
            print(f"로드된 인덱스 타입: {index_type}")

            # 인덱스 정보 출력
            print(f"인덱스 차원: {index.d}, 벡터 수: {index.ntotal}")

            # L2 인덱스를 내적(코사인 유사도) 인덱스로 변환
            if isinstance(index, faiss.IndexFlatL2) or "L2" in index_type:
                print("유클리드 거리 인덱스를 코사인 유사도 인덱스로 변환합니다...")
                try:
                    # 벡터 추출 시도
                    vectors = index.reconstruct_n(0, index.ntotal)

                    # 벡터 통계 확인
                    print(f"추출된 벡터 형태: {vectors.shape}")
                    print(
                        f"벡터 통계: 최소값={np.min(vectors)}, 최대값={np.max(vectors)}"
                    )

                    # 벡터 노름 확인
                    norms = np.linalg.norm(vectors, axis=1)
                    print(
                        f"변환 전 벡터 노름 - 평균: {np.mean(norms)}, 최소: {np.min(norms)}, 최대: {np.max(norms)}"
                    )

                    # 정규화 수행
                    faiss.normalize_L2(vectors)

                    # 정규화 후 노름 확인
                    norms_after = np.linalg.norm(vectors, axis=1)
                    print(
                        f"변환 후 벡터 노름 - 평균: {np.mean(norms_after)}, 최소: {np.min(norms_after)}, 최대: {np.max(norms_after)}"
                    )

                    # 새 내적 인덱스 생성
                    new_index = faiss.IndexFlatIP(index.d)
                    new_index.add(vectors)

                    print(
                        f"변환 완료: {index.ntotal}개 벡터가 코사인 유사도 인덱스로 변환됨"
                    )
                    index = new_index
                except Exception as e:
                    print(f"인덱스 변환 중 오류: {e}")
                    print("원본 인덱스를 계속 사용합니다.")

            # 메타데이터 파일 로드 및 디버깅
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

            # 메타데이터 형식 분석
            metadata_count = len(metadata)
            sample_keys = list(metadata.keys())[:5]
            print(f"메타데이터 항목 수: {metadata_count}")
            print(f"메타데이터 샘플 키: {sample_keys}")
            if sample_keys:
                key_types = [type(k) for k in sample_keys]
                print(f"메타데이터 키 타입: {key_types}")

                # 첫 번째 메타데이터 항목 내용 확인
                first_item = metadata[sample_keys[0]]
                print(
                    f"메타데이터 첫 항목: {first_item.keys() if isinstance(first_item, dict) else 'Not a dict'}"
                )

            self.collections.append(
                {"name": collection_name, "index": index, "metadata": metadata}
            )
            print(f"{collection_name} 컬렉션 로드 완료: {len(metadata)}개 벡터")
            return True
        except Exception as e:
            print(f"{collection_name} 컬렉션 로드 중 오류: {e}")
            return False

    def get_upstage_embedding(self, text):
        if text in self.cached_embeddings:
            return self.cached_embeddings[text]

        if not self.api_key or len(self.api_key) < 10:
            raise ValueError(
                f"유효한 Upstage API 키가 없습니다. 현재 키: {self.api_key}"
            )

        try:
            # OpenAI 클라이언트 생성
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key, base_url="https://api.upstage.ai/v1/solar"
            )

            # LangSmith 트래킹이 활성화된 경우 클라이언트 래핑
            if os.getenv("LANGCHAIN_API_KEY"):
                from langsmith.wrappers import wrap_openai

                client = wrap_openai(client)  # 인스턴스를 래핑

            # 임베딩 생성 요청과 트래킹
            with langsmith.trace(
                name="upstage_embedding",
                project_name=LANGCHAIN_PROJECT,
                tags=["insupanda", "embedding"],
                metadata={"text_length": len(text)},
            ) as run:
                response = client.embeddings.create(input=text, model="embedding-query")
                embedding = response.data[0].embedding

                print(f"임베딩 생성 성공, 차원: {len(embedding)}")

                vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
                # 원본 벡터 차원 저장
                original_dim = vector.shape[1]

                # 정규화 이전 벡터 노름 계산
                norm_before = np.linalg.norm(vector)
                print(f"정규화 전 벡터 노름: {norm_before}")

                # L2 정규화 수행
                faiss.normalize_L2(vector)

                # 정규화 이후 벡터 노름 확인 (항상 1에 가까워야 함)
                norm_after = np.linalg.norm(vector)
                print(f"정규화 후 벡터 노름: {norm_after}")

                # 벡터 통계치 확인
                print(
                    f"벡터 통계: 최소값={np.min(vector)}, 최대값={np.max(vector)}, 평균={np.mean(vector)}"
                )
                print(f"내적 기반 검색을 위해 준비된 임베딩 형태: {vector.shape}")

                # 결과 캐싱 및 메타데이터 추가
                self.cached_embeddings[text] = vector
                if run:
                    run.add_metadata(
                        {
                            "embedding_dimension": original_dim,
                            "norm_before": float(norm_before),
                            "norm_after": float(norm_after),
                            "success": True,
                        }
                    )

                return vector
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            if os.getenv("LANGCHAIN_API_KEY"):
                with langsmith.trace(
                    name="embedding_error",
                    project_name=LANGCHAIN_PROJECT,
                    tags=["insupanda", "error", "embedding"],
                    metadata={"error": str(e)},
                ) as error_run:
                    if error_run:
                        error_run.add_metadata(
                            {"error_message": f"임베딩 생성 오류: {str(e)}"}
                        )
            raise ValueError(
                "임베딩 생성에 실패했습니다. API 키와 네트워크 상태를 확인하세요."
            )

    def search(self, query, collection_names=None, top_k=2):
        if not self.collections:
            return [
                {
                    "collection": "default",
                    "id": "0",
                    "score": 1.0,
                    "metadata": {"text": "로드된 컬렉션이 없습니다."},
                }
            ]

        all_results = []
        use_collections = [
            c
            for c in self.collections
            if not collection_names or c["name"] in collection_names
        ]
        if not use_collections:
            return [
                {
                    "collection": "default",
                    "id": "0",
                    "score": 1.0,
                    "metadata": {"text": "지정된 컬렉션을 찾을 수 없습니다."},
                }
            ]

        print(f"\n-------- 벡터 검색 시작 --------")
        print(f"쿼리: '{query}'")
        print(f"대상 컬렉션: {[c['name'] for c in use_collections]}")
        print(f"각 컬렉션당 top_k: {top_k}")

        try:
            # LangSmith로 임베딩 생성 및 검색 과정 트래킹
            with langsmith.trace(
                name="vector_search",
                project_name=LANGCHAIN_PROJECT,
                tags=["insupanda", "vector_search"],
                metadata={
                    "query": query,
                    "collections": collection_names,
                    "top_k": top_k,
                },
            ) as run:
                query_embedding = self.get_upstage_embedding(query)
                # 임베딩 형태 출력
                print(
                    f"쿼리 임베딩 형태: {type(query_embedding)}, 타입: {query_embedding.dtype}"
                )
                if isinstance(query_embedding, list):
                    query_embedding = np.array(
                        query_embedding, dtype=np.float32
                    ).reshape(1, -1)
                    print(f"리스트에서 배열로 변환됨: {query_embedding.shape}")

                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.reshape(1, -1)
                    print(f"1D 배열에서 2D로 변환됨: {query_embedding.shape}")

                query_dim = query_embedding.shape[1]
                print(f"쿼리 임베딩 차원: {query_dim}")

                # 각 컬렉션에서 항상 top_k개의 문서 검색
                for collection in use_collections:
                    try:
                        index = collection["index"]
                        metadata = collection["metadata"]
                        collection_name = collection["name"]

                        print(f"\n검색 중: {collection_name} 컬렉션")
                        print(f"인덱스 차원: {index.d}, 인덱스 타입: {type(index)}")
                        print(
                            f"메타데이터 타입: {type(metadata)}, 항목 수: {len(metadata)}"
                        )

                        if query_dim != index.d:
                            print(f"차원 불일치: 쿼리={query_dim}, 인덱스={index.d}")
                            # 차원이 다른 경우 벡터를 올바른 차원으로 패딩하거나 자름
                            if query_dim < index.d:
                                # 패딩: 부족한 차원을 0으로 채움
                                padded = np.zeros((1, index.d), dtype=np.float32)
                                padded[0, :query_dim] = query_embedding[0, :]
                                query_embedding = padded
                                print(
                                    f"쿼리 벡터를 {query_dim}에서 {index.d}로 패딩했습니다."
                                )
                            else:
                                # 자름: 여분의 차원을 제거
                                query_embedding = query_embedding[0, : index.d].reshape(
                                    1, -1
                                )
                                print(
                                    f"쿼리 벡터를 {query_dim}에서 {index.d}로 잘랐습니다."
                                )

                        # L2 정규화 다시 적용
                        faiss.normalize_L2(query_embedding)

                        # 검색 전 벡터 상태 로깅
                        query_norm = np.linalg.norm(query_embedding)
                        print(f"검색 전 쿼리 벡터 노름: {query_norm}")

                        # 노름이 1과 크게 차이나면 경고
                        if abs(query_norm - 1.0) > 1e-5:
                            print(f"경고: 쿼리 벡터 노름이 1이 아닙니다: {query_norm}")
                            # 강제로 정규화
                            query_embedding = query_embedding / query_norm
                            print(
                                f"강제 정규화 후 노름: {np.linalg.norm(query_embedding)}"
                            )

                        print(f"검색 쿼리 차원: {query_embedding.shape}")

                        # 각 컬렉션에서 항상 top_k개의 문서 검색
                        distances, indices = index.search(query_embedding, top_k)

                        # 내적 값이 1보다 크면 경고
                        if np.any(distances > 1.01):  # 약간의 오차 허용
                            print(
                                f"경고: 내적 값이 1보다 큽니다. 최댓값: {np.max(distances)}"
                            )
                            # 내적 값이 1을 초과하는 경우 1로 제한
                            distances = np.minimum(distances, 1.0)

                        # 내적 기반 검색에서는 값이 클수록 유사도가 높음
                        print(f"검색 결과: 내적 값={distances}, 인덱스={indices}")

                        # 유사도 점수 변환 (내적 값은 -1~1 범위, 높을수록 유사)
                        # 결과를 0~1 범위로 정규화 (옵션)
                        normalized_scores = (distances + 1) / 2

                        collection_results = []
                        for i, (idx, score) in enumerate(
                            zip(indices[0], normalized_scores[0])
                        ):
                            if idx != -1:  # -1은 결과가 없음을 의미
                                # 메타데이터에서 해당 인덱스의 정보 가져오기
                                doc_id = str(idx)

                                # 디버깅: 메타데이터 키 확인
                                print(f"메타데이터 키 검색: {doc_id}")
                                print(f"메타데이터 키 타입: {type(doc_id)}")

                                # 메타데이터 키가 존재하는지 확인
                                if doc_id in metadata:
                                    print(f"메타데이터에서 키 {doc_id} 찾음")
                                    doc_metadata = metadata[doc_id]
                                    # 결과 추가 (점수는 높을수록 유사함을 의미)
                                    collection_results.append(
                                        {
                                            "collection": collection_name,
                                            "id": doc_id,
                                            "score": float(
                                                score
                                            ),  # 0~1 사이 값, 높을수록 유사
                                            "metadata": doc_metadata,
                                        }
                                    )
                                else:
                                    # 키가 존재하지 않으면 다른 형식으로 시도
                                    print(
                                        f"메타데이터에서 키 {doc_id} 찾을 수 없음. 다른 형식 시도..."
                                    )

                                    # 정수 키로 시도
                                    int_id = str(int(idx))
                                    if int_id in metadata:
                                        print(f"정수 키 {int_id}로 찾음")
                                        doc_metadata = metadata[int_id]
                                        collection_results.append(
                                            {
                                                "collection": collection_name,
                                                "id": int_id,
                                                "score": float(score),
                                                "metadata": doc_metadata,
                                            }
                                        )
                                    else:
                                        # 인덱스가 리스트의 범위 내에 있는지 확인
                                        try:
                                            # 메타데이터가 실제로는 리스트이거나 인덱스 기반인 경우
                                            idx_int = int(idx)
                                            meta_len = len(metadata)
                                            if 0 <= idx_int < meta_len:
                                                print(
                                                    f"인덱스 {idx_int}를 배열 접근으로 시도 (배열 길이: {meta_len})"
                                                )
                                                try:
                                                    # 리스트로 변환된 딕셔너리에서 키로 접근
                                                    list_keys = list(metadata.keys())
                                                    if idx_int < len(list_keys):
                                                        key = list_keys[idx_int]
                                                        doc_metadata = metadata[key]
                                                        print(f"변환된 키 {key}로 찾음")
                                                        collection_results.append(
                                                            {
                                                                "collection": collection_name,
                                                                "id": key,
                                                                "score": float(score),
                                                                "metadata": doc_metadata,
                                                            }
                                                        )
                                                        continue
                                                except Exception as e:
                                                    print(f"키 변환 접근 오류: {e}")
                                        except Exception as e:
                                            print(f"인덱스 기반 접근 시도 중 오류: {e}")

                                        # 메타데이터 키 출력 (처음 10개만)
                                        meta_keys = list(metadata.keys())[:10]
                                        print(f"메타데이터 샘플 키: {meta_keys}")

                                        # 모든 방법 실패 시 기본 메타데이터 생성
                                        print(
                                            f"모든 키 검색 실패, 기본 메타데이터 사용"
                                        )
                                        collection_results.append(
                                            {
                                                "collection": collection_name,
                                                "id": doc_id,
                                                "score": float(score),
                                                "metadata": {
                                                    "text": f"인덱스 {idx}의 메타데이터를 찾을 수 없습니다."
                                                },
                                            }
                                        )
                                        # 이 결과도 all_results에 추가
                                        all_results.append(
                                            {
                                                "collection": collection_name,
                                                "id": doc_id,
                                                "score": float(score),
                                                "metadata": {
                                                    "text": f"인덱스 {idx}의 메타데이터를 찾을 수 없습니다."
                                                },
                                            }
                                        )

                        # all_results에 collection_results 추가
                        all_results.extend(collection_results)

                        print(
                            f"{collection_name} 컬렉션에서 {len(collection_results)}개 결과 찾음"
                        )
                        # 각 청크의 시작 부분 미리보기 출력
                        for i, result in enumerate(collection_results):
                            text = result.get("metadata", {}).get("text", "")
                            preview = text[:100] + "..." if text else "텍스트 없음"
                            print(
                                f"  청크 {i+1}: 점수={result['score']:.4f}, 미리보기: {preview}"
                            )
                    except Exception as e:
                        print(f"컬렉션 '{collection_name}' 검색 중 오류: {e}")
                        import traceback

                        print(f"상세 오류 정보: {traceback.format_exc()}")
                        if run:
                            run.add_metadata(
                                {
                                    "error": f"컬렉션 '{collection_name}' 검색 오류: {str(e)}"
                                }
                            )
                        continue

                # 결과를 점수에 따라 정렬
                all_results.sort(key=lambda x: x["score"])

                print(f"\n총 {len(all_results)}개 청크 검색됨")
                print(f"-------- 벡터 검색 완료 --------\n")

                # 트레이싱 메타데이터 업데이트
                if run:
                    run.add_metadata(
                        {
                            "result_count": len(all_results),
                            "collections_searched": [
                                c["name"] for c in use_collections
                            ],
                        }
                    )

                return (
                    all_results
                    if all_results
                    else [
                        {
                            "collection": "default",
                            "id": "0",
                            "score": 1.0,
                            "metadata": {"text": "검색 결과를 찾을 수 없습니다."},
                        }
                    ]
                )

        except Exception as e:
            print(f"벡터 검색 중 오류: {e}")
            if os.getenv("LANGCHAIN_API_KEY"):
                with langsmith.trace(
                    name="vector_search_error",
                    project_name=LANGCHAIN_PROJECT,
                    tags=["insupanda", "error"],
                    metadata={"error": str(e), "query": query},
                ) as error_run:
                    if error_run:
                        error_run.add_metadata(
                            {"error_message": f"검색 오류: {str(e)}"}
                        )
            return [
                {
                    "collection": "default",
                    "id": "0",
                    "score": 1.0,
                    "metadata": {"text": f"검색 중 오류 발생: {str(e)}"},
                }
            ]

    def generate_answer(self, query, search_results, openai_api_key):
        if not search_results:
            return "검색 결과가 없습니다. 다른 질문을 시도해보세요."
        if not openai_api_key:
            return "OpenAI API key가 제공되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정해주세요."

        print(f"\n-------- 답변 생성 시작 --------")
        print(f"질문: '{query}'")
        print(f"검색 결과 수: {len(search_results)}")

        # 검색 결과를 보험사별로 그룹화
        company_results = {}
        for result in search_results:
            collection_name = result.get("collection", "")
            company_name = self.collection_to_company_mapping.get(collection_name, collection_name)
            if company_name not in company_results:
                company_results[company_name] = []
            company_results[company_name].append(result)

        print(f"검색된 보험사 수: {len(company_results)}")
        for company, results in company_results.items():
            print(f"- {company}: {len(results)}개 청크")

        # 보험사별 컨텍스트 생성
        context = ""
        multiple_companies = len(company_results) > 1

        if multiple_companies:
            print(f"\n여러 보험사 비교 컨텍스트 생성 중...")
        else:
            print(f"\n단일 보험사 컨텍스트 생성 중...")

        for company, results in company_results.items():
            company_context = ""

            if multiple_companies:
                company_context += f"\n\n## {company} 정보:\n"

            for result in results:
                text = result.get("metadata", {}).get("text", "")
                if text:
                    company_context += f"\n---\n{text}"

            context += company_context
            text_preview = (
                company_context[:150] + "..."
                if len(company_context) > 150
                else company_context
            )
            print(f"{company} 컨텍스트: {text_preview}")

        if not context:
            return "관련 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시거나, 다른 키워드를 사용해보세요."

        # 비교 요청인지 감지
        query_lower = query.lower()
        comparison_keywords = [
            "비교",
            "차이",
            "다른",
            "다른점",
            "비교해",
            "비교해줘",
            "차이점",
            "알려줘",
            "뭐가 더 나은가",
        ]
        detected_comparison_keywords = [
            kw for kw in comparison_keywords if kw in query_lower
        ]
        is_comparison = len(detected_comparison_keywords) > 0

        if is_comparison:
            print(f"비교 질문 감지: {detected_comparison_keywords}")

        system_message = "You are an insurance policy expert. Always answer in Korean."
        if multiple_companies and is_comparison:
            system_message += " 여러 보험사의 약관을 비교 분석하여 차이점과 공통점을 명확하게 설명해주세요. 표 형식으로 정리해서 답변하면 더 좋습니다."

        prompt = f"""질문: {query}\n\n관련 문서: {context}\n\n답변:"""

        print(f"시스템 메시지: {system_message}")
        print(f"프롬프트 길이: {len(prompt)} 자")

        # LangChain의 ChatOpenAI 모델 초기화 - 최신 버전 호환성 반영
        try:
            # 최신 LangChain API와 호환되도록 수정
            chat = ChatOpenAI(
                api_key=openai_api_key,
                temperature=0.7,
                max_tokens=2000,
                model="gpt-4o-mini",
            )

            # 메타데이터 기록을 위한 정보
            metadata = {
                "query": query,
                "is_comparison": is_comparison,
                "multiple_companies": multiple_companies,
                "company_count": len(company_results),
                "companies": list(company_results.keys()),
                "context_length": len(context),
            }

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt),
            ]

            print(f"LLM 호출 중...")

            # LangSmith 런 생성 및 트래킹 - 최신 API로 업데이트
            try:
                with langsmith.trace(
                    name="generate_insurance_answer",
                    project_name=LANGCHAIN_PROJECT,
                    tags=["insupanda", "llm_response"],
                    metadata=metadata,
                ) as run:
                    response = chat.invoke(messages)
                    answer = response.content
                    print(f"LLM 응답 생성 완료 (길이: {len(answer)} 자)")
                    if run:
                        run.add_metadata({"status": "응답 성공"})

                    print(f"-------- 답변 생성 완료 --------\n")
                    print("answer", answer)
                    
                    # 답변에서 컬렉션 이름을 실제 보험사 이름으로 변환
                    for collection_name, company_name in self.collection_to_company_mapping.items():
                        if collection_name in answer:
                            answer = answer.replace(collection_name, company_name)
                    
                    return answer
            except Exception as e:
                print(f"LangSmith 트래킹 오류: {e}")
                # LangSmith 오류가 있어도 LLM 응답은 계속 진행
                response = chat.invoke(messages)
                answer = response.content
                print(f"LLM 응답 생성 완료 (길이: {len(answer)} 자)")
                print(f"-------- 답변 생성 완료 --------\n")
                print(answer)
                
                # 답변에서 컬렉션 이름을 실제 보험사 이름으로 변환
                for collection_name, company_name in self.collection_to_company_mapping.items():
                    if collection_name in answer:
                        answer = answer.replace(collection_name, company_name)
                
                return answer

        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            print(f"직접 OpenAI API 호출 시도 중...")
            # 최후의 수단으로 직접 OpenAI API 호출 시도
            try:
                from openai import OpenAI

                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
                answer = response.choices[0].message.content
                print(f"OpenAI API 직접 호출 성공 (길이: {len(answer)} 자)")
                print(f"-------- 답변 생성 완료 --------\n")
                
                # 답변에서 컬렉션 이름을 실제 보험사 이름으로 변환
                for collection_name, company_name in self.collection_to_company_mapping.items():
                    if collection_name in answer:
                        answer = answer.replace(collection_name, company_name)
                
                return answer
            except Exception as fallback_error:
                print(f"직접 OpenAI API 호출 오류: {fallback_error}")
                print(f"-------- 답변 생성 실패 --------\n")
                return f"답변 생성 중 오류가 발생했습니다. 관리자에게 문의해주세요. 오류: {str(e)}"

    def find_matching_collections(self, question, available_collections):
        """
        사용자 질문에서 보험사 관련 키워드를 검출하여 일치하는 컬렉션 이름 목록 반환
        비교 질문인 경우 관련된 모든 보험사 컬렉션 반환
        """
        print(f"\n-------- 컬렉션 매칭 시작 --------")
        print(f"질문: '{question}'")
        print(f"사용 가능한 컬렉션: {available_collections}")

        if not question or not available_collections:
            print(f"질문이 비어있거나 사용 가능한 컬렉션이 없음")
            print("-------- 컬렉션 매칭 실패 --------\n")
            return []

        # 정규화된 질문 (소문자, 공백 제거)
        normalized_question = question.lower().replace(" ", "")

        # 보험사 키워드 매핑
        insurance_company_keywords = {
            "삼성화재": ["삼성화재", "삼성", "samsung"],
            "DB손해보험": [
                "db손해보험",
                "db손해",
                "db보험",
                "db",
                "디비손해보험",
                "디비",
            ],
            "하나손해보험": ["하나손해보험", "하나손보", "하나", "hana"],
            "한화손해보험": ["한화손해보험", "한화손보", "한화", "hanwha"],
            "흥국화재": ["흥국화재", "흥국", "heung", "흥국생명"],
            "현대해상": ["현대해상", "현대", "hyundai"],
            "KB손해보험": ["KB손해보험", "KB손보", "KB", "케이비"],
            "롯데손해보험": ["롯데손해보험", "롯데손보", "롯데", "lotte"],
            "MG손해보험": ["MG손해보험", "MG손보", "MG", "엠지"],
            "메리츠화재": ["메리츠화재", "메리츠", "meritz"],
            "NH농협손해보험": [
                "NH농협손해보험",
                "NH손해보험",
                "농협손해보험",
                "NH손보",
                "농협손보",
                "NH",
                "농협",
            ],
        }

        # 보험 종류 키워드
        insurance_type_keywords = [
            "암",
            "상해",
            "질병",
            "재물",
            "화재",
            "운전자",
            "자동차",
            "실손",
        ]

        # 비교 요청 키워드
        comparison_keywords = [
            "비교",
            "차이",
            "다른",
            "다른점",
            "비교해",
            "비교해줘",
            "차이점",
            "알려줘",
            "뭐가 더 나은가",
        ]

        # 언급된 보험사 추적
        mentioned_companies = []
        for company, keywords in insurance_company_keywords.items():
            if any(keyword in normalized_question for keyword in keywords):
                mentioned_companies.append(company)
                print(f"보험사 키워드 감지: {company}")

        # 비교 요청 감지
        is_comparison_request = any(
            keyword in normalized_question for keyword in comparison_keywords
        )
        if is_comparison_request:
            detected_keywords = [
                keyword
                for keyword in comparison_keywords
                if keyword in normalized_question
            ]
            print(f"비교 키워드 감지: {detected_keywords}")

        # 보험 종류 키워드 감지
        detected_insurance_types = [
            keyword
            for keyword in insurance_type_keywords
            if keyword in normalized_question
        ]
        if detected_insurance_types:
            print(f"보험 종류 키워드 감지: {detected_insurance_types}")

        # 컬렉션 매칭 로직
        matched_collections = []

        # 두 개 이상 보험사가 언급되었거나 비교 요청이 있는 경우
        # '암' 키워드가 언급된 경우에도 모든 보험사 정보가 필요할 수 있음
        if (
            len(mentioned_companies) > 1
            or is_comparison_request
            or "암" in detected_insurance_types
        ):
            print(f"다중 보험사 비교 또는 암 관련 질문 감지됨")
            # 모든 보험사 컬렉션 추가
            for collection in available_collections:
                is_relevant = False

                # 삼성화재 컬렉션 매칭
                if "samsung" in collection.lower() or "삼성" in collection:
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"삼성화재 컬렉션 매칭: {collection}")

                # DB손해보험 컬렉션 매칭
                elif (
                    "db" in collection.lower()
                    or "디비" in collection
                    or "sonbo" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"DB손해보험 컬렉션 매칭: {collection}")

                # 하나손해보험 컬렉션 매칭
                elif (
                    "hana" in collection.lower()
                    or "하나" in collection
                    or "ha" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"하나손해보험 컬렉션 매칭: {collection}")

                # 한화손해보험 컬렉션 매칭
                elif (
                    "hanwha" in collection.lower()
                    or "한화" in collection
                    or "hw" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"한화손해보험 컬렉션 매칭: {collection}")

                # 흥국화재 컬렉션 매칭
                elif (
                    "heung" in collection.lower()
                    or "흥국" in collection
                    or "hg" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"흥국화재 컬렉션 매칭: {collection}")

                # 현대해상 컬렉션 매칭
                elif (
                    "hyundai" in collection.lower()
                    or "현대" in collection
                    or "hd" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"현대해상 컬렉션 매칭: {collection}")

                # KB손해보험 컬렉션 매칭
                elif (
                    "kb" in collection.lower()
                    or "KB" in collection
                    or "케이비" in collection
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"KB손해보험 컬렉션 매칭: {collection}")

                # 롯데손해보험 컬렉션 매칭
                elif (
                    "lotte" in collection.lower()
                    or "롯데" in collection
                    or "lt" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"롯데손해보험 컬렉션 매칭: {collection}")

                # MG손해보험 컬렉션 매칭
                elif (
                    "mg" in collection.lower()
                    or "MG" in collection
                    or "엠지" in collection
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"MG손해보험 컬렉션 매칭: {collection}")

                # 메리츠화재 컬렉션 매칭
                elif (
                    "meritz" in collection.lower()
                    or "메리츠" in collection
                    or "mz" in collection.lower()
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"메리츠화재 컬렉션 매칭: {collection}")

                # NH농협손해보험 컬렉션 매칭
                elif (
                    "nh" in collection.lower()
                    or "NH" in collection
                    or "농협" in collection
                ):
                    matched_collections.append(collection)
                    is_relevant = True
                    print(f"NH농협손해보험 컬렉션 매칭: {collection}")

                if not is_relevant and len(mentioned_companies) == 0:
                    # 특정 보험사가 언급되지 않은 경우 모든 컬렉션 추가
                    matched_collections.append(collection)
                    print(f"기본 컬렉션 매칭: {collection}")
        else:
            # 단일 보험사만 언급된 경우
            for collection in available_collections:
                if "삼성화재" in mentioned_companies and (
                    "samsung" in collection.lower() or "삼성" in collection
                ):
                    matched_collections.append(collection)
                    print(f"삼성화재 컬렉션 매칭: {collection}")

                elif "DB손해보험" in mentioned_companies and (
                    "db" in collection.lower()
                    or "디비" in collection
                    or "sonbo" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"DB손해보험 컬렉션 매칭: {collection}")

                elif "하나손해보험" in mentioned_companies and (
                    "hana" in collection.lower()
                    or "하나" in collection
                    or "ha" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"하나손해보험 컬렉션 매칭: {collection}")

                elif "한화손해보험" in mentioned_companies and (
                    "hanwha" in collection.lower()
                    or "한화" in collection
                    or "hw" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"한화손해보험 컬렉션 매칭: {collection}")

                elif "흥국화재" in mentioned_companies and (
                    "heung" in collection.lower()
                    or "흥국" in collection
                    or "hg" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"흥국화재 컬렉션 매칭: {collection}")

                elif "현대해상" in mentioned_companies and (
                    "hyundai" in collection.lower()
                    or "현대" in collection
                    or "hd" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"현대해상 컬렉션 매칭: {collection}")

                elif "KB손해보험" in mentioned_companies and (
                    "kb" in collection.lower()
                    or "KB" in collection
                    or "케이비" in collection
                ):
                    matched_collections.append(collection)
                    print(f"KB손해보험 컬렉션 매칭: {collection}")

                elif "롯데손해보험" in mentioned_companies and (
                    "lotte" in collection.lower()
                    or "롯데" in collection
                    or "lt" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"롯데손해보험 컬렉션 매칭: {collection}")

                elif "MG손해보험" in mentioned_companies and (
                    "mg" in collection.lower()
                    or "MG" in collection
                    or "엠지" in collection
                ):
                    matched_collections.append(collection)
                    print(f"MG손해보험 컬렉션 매칭: {collection}")

                elif "메리츠화재" in mentioned_companies and (
                    "meritz" in collection.lower()
                    or "메리츠" in collection
                    or "mz" in collection.lower()
                ):
                    matched_collections.append(collection)
                    print(f"메리츠화재 컬렉션 매칭: {collection}")

                elif "NH농협손해보험" in mentioned_companies and (
                    "nh" in collection.lower()
                    or "NH" in collection
                    or "농협" in collection
                ):
                    matched_collections.append(collection)
                    print(f"NH농협손해보험 컬렉션 매칭: {collection}")

                # 보험사가 언급되지 않은 경우 모든 컬렉션 추가
                elif len(mentioned_companies) == 0:
                    matched_collections.append(collection)
                    print(f"기본 컬렉션 매칭: {collection}")

        # 중복 제거
        matched_collections = list(set(matched_collections))

        print(f"최종 매칭된 컬렉션: {matched_collections}")
        print(f"-------- 컬렉션 매칭 완료 --------\n")

        return matched_collections

    def create_index(self, embeddings, dimension=1024):
        """임베딩 배열로부터 FAISS 인덱스를 생성합니다."""
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


def response(query: SearchQuery):
    global rag
    # 만약 query가 문자열이면 SearchQuery 객체로 감쌈
    if isinstance(query, str):
        query = SearchQuery(query=query, collections=[])

    available_collections = [
        d
        for d in os.listdir(rag.base_path)
        if os.path.isdir(os.path.join(rag.base_path, d))
    ]
    print(f"사용 가능한 컬렉션: {available_collections}")

    # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾습니다
    use_collections = (
        query.collections
        if query.collections
        else rag.find_matching_collections(query.query_text, available_collections)
    )
    print(f"사용할 컬렉션: {use_collections}")

    # 찾은 컬렉션 로드
    for collection_name in use_collections:
        rag.load_collection(collection_name)

    # 청크 탐색 (각 컬렉션당 top_k=2)
    search_results = rag.search(query.query_text, use_collections, top_k=2)

    # 답변 생성
    answer = rag.generate_answer(
        query.query_text, search_results, os.getenv("OPENAI_API_KEY")
    )

    return answer


def search(query: SearchQuery):
    try:
        global rag
        # 디버깅을 위한 요청 본문 출력
        query_text = query.query_text if hasattr(query, "query_text") else "None"
        print(f"\n============== API 요청 받음 ==============")
        print(f"POST /api/search - 쿼리: '{query_text}'")
        print(f"요청된 컬렉션: {query.collections}")

        # LangSmith에서 전체 API 요청 트래킹
        with langsmith.trace(
            name="search_api_request",
            project_name=LANGCHAIN_PROJECT,
            tags=["insupanda", "api", "post_request"],
            metadata={
                "query": query_text,
                "collections_requested": query.collections,
                "endpoint": "/search (POST)",
            },
        ) as run:
            available_collections = [
                d
                for d in os.listdir(rag.base_path)
                if os.path.isdir(os.path.join(rag.base_path, d))
            ]
            print(f"사용 가능한 컬렉션: {available_collections}")

            # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾습니다
            use_collections = (
                query.collections
                if query.collections
                else rag.find_matching_collections(
                    query.query_text, available_collections
                )
            )
            print(f"사용할 컬렉션: {use_collections}")
            if not use_collections:
                if run:
                    run.add_metadata({"notification": "사용할 컬렉션을 찾을 수 없음"})
                return JSONResponse(
                    content={
                        "answer": "질문에 해당하는 보험사 정보를 찾을 수 없습니다. 보험사 이름이나 보험 종류를 명확히 언급해 주세요.",
                        "collections_used": [],
                        "error": "No collections available",
                    },
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )

            # 찾은 컬렉션 로드
            for collection_name in use_collections:
                rag.load_collection(collection_name)

            # 청크 탐색 (각 컬렉션당 top_k=2)
            search_results = rag.search(query.query_text, use_collections, top_k=2)

            # 답변 생성
            answer = rag.generate_answer(
                query.query_text, search_results, os.getenv("OPENAI_API_KEY")
            )

            # 사용된 컬렉션 목록 반환
            friendly_names = [get_friendly_name(c) for c in use_collections]
            print(f"사용된 컬렉션: {friendly_names}")

            # 메타데이터 업데이트
            if run:
                run.add_metadata(
                    {
                        "collections_used": friendly_names,
                        "search_results_count": len(search_results),
                        "answer_length": len(answer),
                    }
                )

            return JSONResponse(
                content={
                    "answer": answer,
                    "collections_used": friendly_names,
                    "error": "",
                },
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
    except Exception as e:
        print(f"API 오류: {e}")
        if os.getenv("LANGCHAIN_API_KEY"):
            with langsmith.trace(
                name="search_api_error",
                project_name=LANGCHAIN_PROJECT,
                tags=["insupanda", "error", "api"],
                metadata={"error": str(e), "endpoint": "/search (POST)"},
            ) as error_run:
                if error_run:
                    error_run.add_metadata({"error_message": f"API 오류: {str(e)}"})
        return JSONResponse(
            content={
                "answer": f"오류 발생: {str(e)}",
                "collections_used": [],
                "error": str(e),
            },
            headers={"Content-Type": "application/json; charset=utf-8"},
        )


# GET 요청도 처리할 수 있도록 추가
async def search_get(query: str, collections: List[str] = None):
    try:
        global rag
        # 디버깅을 위한 요청 본문 출력
        print(f"\n============== API 요청 받음 ==============")
        print(f"GET /api/search - 쿼리: '{query}'")
        print(f"요청된 컬렉션: {collections}")

        # LangSmith에서 전체 API 요청 트래킹
        with langsmith.trace(
            name="search_api_request",
            project_name=LANGCHAIN_PROJECT,
            tags=["insupanda", "api", "get_request"],
            metadata={
                "query": query,
                "collections_requested": collections,
                "endpoint": "/search (GET)",
            },
        ) as run:
            available_collections = [
                d
                for d in os.listdir(rag.base_path)
                if os.path.isdir(os.path.join(rag.base_path, d))
            ]
            print(f"사용 가능한 컬렉션: {available_collections}")

            # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾습니다
            use_collections = (
                collections
                if collections
                else rag.find_matching_collections(query, available_collections)
            )
            print(f"사용할 컬렉션: {use_collections}")
            if not use_collections:
                if run:
                    run.add_metadata({"notification": "사용할 컬렉션을 찾을 수 없음"})
                return JSONResponse(
                    content={
                        "answer": "질문에 해당하는 보험사 정보를 찾을 수 없습니다. 보험사 이름이나 보험 종류를 명확히 언급해 주세요.",
                        "collections_used": [],
                        "error": "No collections available",
                    },
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )

            # 찾은 컬렉션 로드
            for collection_name in use_collections:
                rag.load_collection(collection_name)

            # 청크 탐색 (각 컬렉션당 top_k=2)
            search_results = rag.search(query, use_collections, top_k=2)

            # 답변 생성
            answer = rag.generate_answer(
                query, search_results, os.getenv("OPENAI_API_KEY")
            )

            # 사용된 컬렉션 목록 반환
            friendly_names = [get_friendly_name(c) for c in use_collections]
            print(f"사용된 컬렉션: {friendly_names}")

            # 메타데이터 업데이트
            if run:
                run.add_metadata(
                    {
                        "collections_used": friendly_names,
                        "search_results_count": len(search_results),
                        "answer_length": len(answer),
                    }
                )

            return JSONResponse(
                content={
                    "answer": answer,
                    "collections_used": friendly_names,
                    "error": "",
                },
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
    except Exception as e:
        print(f"API 오류: {e}")
        if os.getenv("LANGCHAIN_API_KEY"):
            with langsmith.trace(
                name="search_api_error",
                project_name=LANGCHAIN_PROJECT,
                tags=["insupanda", "error", "api"],
                metadata={"error": str(e), "endpoint": "/search (GET)"},
            ) as error_run:
                if error_run:
                    error_run.add_metadata({"error_message": f"API 오류: {str(e)}"})
        return JSONResponse(
            content={
                "answer": f"오류 발생: {str(e)}",
                "collections_used": [],
                "error": str(e),
            },
            headers={"Content-Type": "application/json; charset=utf-8"},
        )


rag = RAGService()
