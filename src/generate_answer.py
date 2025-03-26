from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from typing import List, Dict


def get_friendly_name(collection_name: str) -> str:
    return collection_name


def generate_answer(self, query: str, search_results: List[Dict], openai_api_key: str):
    if not search_results:
        return "검색 결과가 없습니다. 다른 질문을 시도해보세요."
    if not openai_api_key:
        return "OpenAI API key가 제공되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정해주세요."

    print(f"\n-------- 답변 생성 시작 --------")
    print(f"질문: '{query}'")
    print(f"검색 결과 수: {len(search_results)}")

    # 보험사별 그룹화 및 context 생성
    company_results = {}
    for result in search_results:
        collection_name = result.get("collection", "")
        company_results.setdefault(collection_name, []).append(result)

    multiple_companies = len(company_results) > 1
    is_comparison = any(
        kw in query.lower()
        for kw in [
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
    )

    context = ""
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

    if not context.strip():
        return "관련 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시거나, 다른 키워드를 사용해보세요."

    # 시스템 메시지 구성
    system_prompt = "You are an insurance policy expert. Always answer in Korean."
    if multiple_companies and is_comparison:
        system_prompt += " 여러 보험사의 약관을 비교 분석하여 차이점과 공통점을 명확하게 설명해주세요. 표 형식으로 정리하면 좋습니다."

    print(f"🧠 시스템 프롬프트: {system_prompt[:80]}...")
    print(f"📄 문맥 길이: {len(context)}")

    # LCEL 스타일 체인 구성
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            ("human", "질문: {query}\n\n관련 문서: {context}\n\n답변:"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini", api_key=openai_api_key, temperature=0.7, max_tokens=2000
    )

    # LCEL 표현: prompt → llm
    chain: Runnable = prompt | llm

    # 실행
    print("🔍 LLM 응답 생성 중...")
    response = chain.invoke({"query": query, "context": context})

    answer = response.content
    print(f"✅ 응답 완료 (길이: {len(answer)} 자)")
    print(f"-------- 답변 생성 완료 --------\n")
    return answer
