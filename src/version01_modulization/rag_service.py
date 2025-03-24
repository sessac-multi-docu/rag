import os
import langsmith
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
from models import EmbeddingModel, get_embedding_model
from vector_store import VectorStore

class NestedQuery(BaseModel):
    query: str

class SearchQuery(BaseModel):
    query: str
    collections: List[str] = None

    class Config:
        # 추가 유효성 검사와 예제 값 설정
        schema_extra = {
            "example": {
                "query": "삼성화재 암보험에 대해 알려줘",
                "collections": ["Samsung_YakMu2404103NapHae20250113"],
            }
        }

def get_friendly_name(collection_name: str) -> str:
    """컬렉션 이름을 사용자 친화적인 이름으로 변환"""
    # 필요한 경우 여기에 매핑 로직 추가
    return collection_name

class RAGService:
    """RAG(Retrieval Augmented Generation) 서비스 클래스"""
    
    def __init__(
        self,
        embedding_model_type: str = "upstage",
        llm_model: str = None,
        api_key: str = None,
    ):
        """
        RAG 서비스 초기화
        
        Args:
            embedding_model_type: 사용할 임베딩 모델 유형 (기본값: "upstage")
            llm_model: 사용할 LLM 모델 이름 (기본값: config.DEFAULT_LLM_MODEL)
            api_key: API 키 (기본값: 환경 변수에서 로드)
        """
        self.api_key = api_key or config.UPSTAGE_API_KEY
        self.llm_model = llm_model or config.DEFAULT_LLM_MODEL
        
        # 임베딩 모델 초기화
        self.embedding_model = get_embedding_model(
            model_type=embedding_model_type,
            api_key=self.api_key
        )
        
        # 벡터 저장소 초기화
        self.vector_store = VectorStore()
    
    def find_matching_collections(
        self, 
        question: str, 
        available_collections: List[str]
    ) -> List[str]:
        """
        사용자 질문에서 보험사 관련 키워드를 검출하여 일치하는 컬렉션 이름 목록 반환
        비교 질문인 경우 관련된 모든 보험사 컬렉션 반환
        
        Args:
            question: 사용자 질문
            available_collections: 사용 가능한 컬렉션 목록
            
        Returns:
            매칭된 컬렉션 이름 목록
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

                # 보험사가 언급되지 않은 경우 모든 컬렉션 추가
                elif len(mentioned_companies) == 0:
                    matched_collections.append(collection)
                    print(f"기본 컬렉션 매칭: {collection}")

        # 중복 제거
        matched_collections = list(set(matched_collections))

        print(f"최종 매칭된 컬렉션: {matched_collections}")
        print(f"-------- 컬렉션 매칭 완료 --------\n")

        return matched_collections
    
    def search(
        self, 
        query: str, 
        collection_names: List[str] = None, 
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        쿼리 텍스트로 벡터 검색 수행
        
        Args:
            query: 검색 쿼리 텍스트
            collection_names: 검색 대상 컬렉션 이름 목록
            top_k: 각 컬렉션에서 검색할 결과 수
            
        Returns:
            검색 결과 목록
        """
        if not query:
            return [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "검색 쿼리가 비어 있습니다."}
            }]
            
        print(f"\n-------- 벡터 검색 시작 --------")
        print(f"쿼리: '{query}'")
        print(f"대상 컬렉션: {collection_names}")
        print(f"각 컬렉션당 top_k: {top_k}")
        
        try:
            with langsmith.trace(
                name="vector_search",
                project_name=config.LANGCHAIN_PROJECT,
                tags=["insupanda", "vector_search"],
                metadata={
                    "query": query,
                    "collections": collection_names,
                    "top_k": top_k,
                },
            ) as run:
                # 쿼리 임베딩 생성
                query_embedding = self.embedding_model.get_embedding(query)
                
                # 벡터 검색 수행
                search_results = self.vector_store.search(
                    query_vector=query_embedding,
                    collection_names=collection_names,
                    top_k=top_k
                )
                
                # 트레이싱 메타데이터 업데이트
                if run:
                    run.add_metadata({
                        "result_count": len(search_results),
                        "collections_searched": collection_names,
                    })
                
                return search_results
                
        except Exception as e:
            print(f"벡터 검색 중 오류: {e}")
            if config.LANGCHAIN_API_KEY:
                with langsmith.trace(
                    name="vector_search_error",
                    project_name=config.LANGCHAIN_PROJECT,
                    tags=["insupanda", "error"],
                    metadata={"error": str(e), "query": query},
                ) as error_run:
                    if error_run:
                        error_run.add_metadata({"error_message": f"검색 오류: {str(e)}"})
            
            return [{
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": f"검색 중 오류 발생: {str(e)}"}
            }]
    
    def generate_answer(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        openai_api_key: str = None
    ) -> str:
        """
        검색 결과를 기반으로 응답 생성
        
        Args:
            query: 사용자 질문
            search_results: 검색 결과 목록
            openai_api_key: OpenAI API 키 (기본값: 환경 변수에서 로드)
            
        Returns:
            생성된 응답 텍스트
        """
        openai_api_key = openai_api_key or config.OPENAI_API_KEY
        
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
            if collection_name not in company_results:
                company_results[collection_name] = []
            company_results[collection_name].append(result)

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
            friendly_name = get_friendly_name(company)
            company_context = ""

            if multiple_companies:
                company_context += f"\n\n## {friendly_name} 정보:\n"

            for result in results:
                text = result.get("metadata", {}).get("text", "")
                if text:
                    company_context += f"\n---\n{text}"

            context += company_context
            text_preview = company_context[:150] + "..." if len(company_context) > 150 else company_context
            print(f"{friendly_name} 컨텍스트: {text_preview}")

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
        detected_comparison_keywords = [kw for kw in comparison_keywords if kw in query_lower]
        is_comparison = len(detected_comparison_keywords) > 0

        if is_comparison:
            print(f"비교 질문 감지: {detected_comparison_keywords}")

        # 시스템 메시지 구성
        system_message = "You are an insurance policy expert. Always answer in Korean."
        if multiple_companies and is_comparison:
            system_message += " 여러 보험사의 약관을 비교 분석하여 차이점과 공통점을 명확하게 설명해주세요. 표 형식으로 정리해서 답변하면 더 좋습니다."

        # 프롬프트 구성
        prompt = f"""질문: {query}\n\n관련 문서: {context}\n\n답변:"""

        print(f"시스템 메시지: {system_message}")
        print(f"프롬프트 길이: {len(prompt)} 자")

        # LangChain의 ChatOpenAI 모델 호출
        try:
            # 최신 LangChain API와 호환되도록 수정
            chat = ChatOpenAI(
                api_key=openai_api_key,
                temperature=0.7,
                max_tokens=2000,
                model=self.llm_model,
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

            # LangSmith 런 생성 및 트래킹
            try:
                with langsmith.trace(
                    name="generate_insurance_answer",
                    project_name=config.LANGCHAIN_PROJECT,
                    tags=["insupanda", "llm_response"],
                    metadata=metadata,
                ) as run:
                    response = chat.invoke(messages)
                    answer = response.content
                    print(f"LLM 응답 생성 완료 (길이: {len(answer)} 자)")
                    if run:
                        run.add_metadata({"status": "응답 성공"})

                    print(f"-------- 답변 생성 완료 --------\n")
                    return answer
            except Exception as e:
                print(f"LangSmith 트래킹 오류: {e}")
                # LangSmith 오류가 있어도 LLM 응답은 계속 진행
                response = chat.invoke(messages)
                answer = response.content
                print(f"LLM 응답 생성 완료 (길이: {len(answer)} 자)")
                print(f"-------- 답변 생성 완료 --------\n")
                return answer

        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            print(f"직접 OpenAI API 호출 시도 중...")
            # 최후의 수단으로 직접 OpenAI API 호출 시도
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=self.llm_model,
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
                return answer
            except Exception as fallback_error:
                print(f"직접 OpenAI API 호출 오류: {fallback_error}")
                print(f"-------- 답변 생성 실패 --------\n")
                return f"답변 생성 중 오류가 발생했습니다. 관리자에게 문의해주세요. 오류: {str(e)}" 