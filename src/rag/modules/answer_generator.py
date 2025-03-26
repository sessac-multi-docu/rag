"""
LLM을 사용한 답변 생성을 담당하는 모듈
"""
import os
from typing import Dict, List, Any, Optional
import openai
from langsmith import traceable
from .insurance_mappings import InsuranceMappings


class AnswerGenerator:
    """LLM을 활용해 검색 결과로부터 답변을 생성하는 클래스"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        AnswerGenerator 초기화
        
        Args:
            openai_api_key: OpenAI API 키
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.insurance_mappings = InsuranceMappings()
        
        # 환경 변수 설정 확인
        self.use_langsmith = all([
            os.getenv("LANGCHAIN_API_KEY"),
            os.getenv("LANGCHAIN_PROJECT"),
            os.getenv("LANGCHAIN_ENDPOINT")
        ])
    
    @traceable(name="generate_answer")
    def generate_answer(
        self, 
        query_text: str, 
        search_results: Dict[str, List[Dict[str, Any]]],
        model: str = "gpt-4-turbo"
    ) -> str:
        """
        검색 결과를 기반으로 답변 생성
        
        Args:
            query_text: 사용자 질문
            search_results: 검색 결과 딕셔너리
            model: 사용할 OpenAI 모델
            
        Returns:
            생성된 답변
        """
        # 검색 결과가 없는 경우
        if not search_results:
            return "죄송합니다. 해당 질문에 관련된 정보를 찾지 못했습니다."
        
        # 컨텍스트 구성
        context = self._prepare_context(search_results)
        
        # 프롬프트 구성
        prompt = self._create_prompt(query_text, context)
        
        try:
            # LLM 호출
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": 
                     "너는 보험 약관에 대한 질문에 답변하는 보험 전문가야. 주어진 보험 약관 정보를 바탕으로 정확하게 답변해줘. "
                     "답변에 사용한 보험사와 약관 정보를, 답변 마지막에 출처로 명시해줘."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800,
            )
            
            answer = response.choices[0].message.content
            
            # 컬렉션 이름을 실제 보험사 이름으로 변환
            answer = self._replace_collection_names_with_company_names(answer)
            
            return answer
            
        except Exception as e:
            print(f"답변 생성 중 오류 발생: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _prepare_context(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        검색 결과를 LLM 프롬프트용 컨텍스트로 변환
        
        Args:
            search_results: 검색 결과 딕셔너리
            
        Returns:
            포맷팅된 컨텍스트
        """
        context = ""
        
        for collection_name, results in search_results.items():
            company_name = self.insurance_mappings.get_company_name(collection_name)
            context += f"\n===== {company_name} 보험약관 정보 =====\n"
            
            for i, result in enumerate(results):
                content = result.get("content", "")
                source = result.get("source", "출처 미상")
                
                context += f"[정보 {i+1}] 출처: {source}\n"
                context += f"{content}\n\n"
        
        return context
    
    def _create_prompt(self, query_text: str, context: str) -> str:
        """
        LLM 프롬프트 생성
        
        Args:
            query_text: 사용자 질문
            context: 검색 결과 컨텍스트
            
        Returns:
            생성된 프롬프트
        """
        return f"""
다음은 사용자의 보험 약관 관련 질문입니다:
{query_text}

아래는 관련된 보험 약관 정보입니다:
{context}

위 보험 약관 정보를 바탕으로 사용자 질문에 답변해주세요. 약관에 나온 내용만 사용하고, 알 수 없는 내용은 솔직히 모른다고 얘기해주세요.
답변 마지막에는 참고한 보험사와 약관 정보를 출처로 명시해주세요.
"""
    
    def _replace_collection_names_with_company_names(self, text: str) -> str:
        """
        텍스트 내의 컬렉션 이름을 실제 보험사 이름으로 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            변환된 텍스트
        """
        for collection_name, company_name in self.insurance_mappings.collection_to_company.items():
            text = text.replace(collection_name, company_name)
        return text
