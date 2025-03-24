from fastapi import FastAPI, APIRouter, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import langsmith

import config
from rag_service import RAGService, SearchQuery, get_friendly_name

# FastAPI 앱 및 라우터 초기화
app = FastAPI(title="InsuPanda API", description="보험 약관 검색 API")
api_router = APIRouter(prefix="/api")

# RAG 서비스 인스턴스 생성
rag_service = RAGService()

@api_router.post("/search")
async def search(query: SearchQuery):
    """
    벡터 검색 및 응답 생성 API 엔드포인트 (POST)
    
    Args:
        query: 검색 쿼리 객체 (쿼리 텍스트 및 컬렉션 이름 목록)
        
    Returns:
        JSON 응답 (답변, 사용된 컬렉션, 오류 메시지)
    """
    try:
        # 디버깅을 위한 요청 본문 출력
        print(f"\n============== API 요청 받음 ==============")
        print(f"POST /api/search - 쿼리: '{query.query}'")
        print(f"요청된 컬렉션: {query.collections}")

        # LangSmith에서 전체 API 요청 트래킹
        with langsmith.trace(
            name="search_api_request",
            project_name=config.LANGCHAIN_PROJECT,
            tags=["insupanda", "api", "post_request"],
            metadata={
                "query": query.query,
                "collections_requested": query.collections,
                "endpoint": "/search (POST)",
            },
        ) as run:
            # 사용 가능한 컬렉션 목록 가져오기
            available_collections = rag_service.vector_store.get_available_collections()
            print(f"사용 가능한 컬렉션: {available_collections}")

            # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾습니다
            use_collections = (
                query.collections
                if query.collections
                else rag_service.find_matching_collections(
                    query.query, available_collections
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
                rag_service.vector_store.load_collection(collection_name)

            # 청크 탐색 (각 컬렉션당 top_k=2)
            search_results = rag_service.search(query.query, use_collections, top_k=2)

            # 답변 생성
            answer = rag_service.generate_answer(
                query.query, search_results, config.OPENAI_API_KEY
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
        if config.LANGCHAIN_API_KEY:
            with langsmith.trace(
                name="search_api_error",
                project_name=config.LANGCHAIN_PROJECT,
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


@api_router.get("/search")
async def search_get(query: str, collections: Optional[List[str]] = None):
    """
    벡터 검색 및 응답 생성 API 엔드포인트 (GET)
    
    Args:
        query: 검색 쿼리 텍스트
        collections: 검색 대상 컬렉션 이름 목록
        
    Returns:
        JSON 응답 (답변, 사용된 컬렉션, 오류 메시지)
    """
    try:
        # 디버깅을 위한 요청 본문 출력
        print(f"\n============== API 요청 받음 ==============")
        print(f"GET /api/search - 쿼리: '{query}'")
        print(f"요청된 컬렉션: {collections}")

        # LangSmith에서 전체 API 요청 트래킹
        with langsmith.trace(
            name="search_api_request",
            project_name=config.LANGCHAIN_PROJECT,
            tags=["insupanda", "api", "get_request"],
            metadata={
                "query": query,
                "collections_requested": collections,
                "endpoint": "/search (GET)",
            },
        ) as run:
            # 사용 가능한 컬렉션 목록 가져오기
            available_collections = rag_service.vector_store.get_available_collections()
            print(f"사용 가능한 컬렉션: {available_collections}")

            # 요청된 컬렉션이 있거나 쿼리 기반으로 컬렉션을 찾습니다
            use_collections = (
                collections
                if collections
                else rag_service.find_matching_collections(query, available_collections)
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
                rag_service.vector_store.load_collection(collection_name)

            # 청크 탐색 (각 컬렉션당 top_k=2)
            search_results = rag_service.search(query, use_collections, top_k=2)

            # 답변 생성
            answer = rag_service.generate_answer(
                query, search_results, config.OPENAI_API_KEY
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
        if config.LANGCHAIN_API_KEY:
            with langsmith.trace(
                name="search_api_error",
                project_name=config.LANGCHAIN_PROJECT,
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

# 라우터 등록
app.include_router(api_router) 