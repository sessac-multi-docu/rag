"""
보험 상품 비교 기능을 담당하는 모듈
"""
import os
from typing import List, Dict, Any, Optional
import openai
from .insurance_mappings import InsuranceMappings


class CompareService:
    """보험사 및 상품 비교 서비스"""
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        CompareService 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            model: 사용할 모델명
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.insurance_mappings = InsuranceMappings()
    
    def compare_insurance(self, query: str, company_info: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        보험사 및 상품 비교 수행
        
        Args:
            query: 비교 질문
            company_info: 보험사별 정보 딕셔너리
            
        Returns:
            비교 결과
        """
        # 비교 컨텍스트 구성
        context = self._prepare_comparison_context(company_info)
        
        # 비교 프롬프트 구성
        prompt = self._create_comparison_prompt(query, context)
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": 
                     "너는 보험 상품을 객관적으로 비교 분석하는 전문가야. 주어진 보험사들의 정보를 바탕으로 질문에 답변해. "
                     "보험사별 장단점, 보장 내용, 보험료, 특약 등을 명확하게 비교하고, 객관적인 분석을 제공해. "
                     "중요한 차이점을 표로 정리해서 보여주면 좋아. "
                     "답변 마지막에는 정보의 출처를 명시해."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            
            comparison_result = response.choices[0].message.content
            
            # 컬렉션 이름을 실제 보험사 이름으로 변환
            for collection_name, company_name in self.insurance_mappings.collection_to_company.items():
                comparison_result = comparison_result.replace(collection_name, company_name)
            
            return comparison_result
            
        except Exception as e:
            print(f"보험 비교 중 오류 발생: {str(e)}")
            return f"보험 비교 중 오류가 발생했습니다: {str(e)}"
    
    def _prepare_comparison_context(self, company_info: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        비교 컨텍스트 구성
        
        Args:
            company_info: 보험사별 정보 딕셔너리
            
        Returns:
            비교용 컨텍스트
        """
        context = ""
        
        for collection_name, results in company_info.items():
            company_name = self.insurance_mappings.get_company_name(collection_name)
            context += f"\n===== {company_name} 정보 =====\n"
            
            for i, result in enumerate(results):
                content = result.get("content", "")
                source = result.get("source", "출처 미상")
                
                context += f"[정보 {i+1}] 출처: {source}\n"
                context += f"{content}\n\n"
        
        return context
    
    def _create_comparison_prompt(self, query: str, context: str) -> str:
        """
        비교 프롬프트 구성
        
        Args:
            query: 비교 질문
            context: 비교 컨텍스트
            
        Returns:
            비교 프롬프트
        """
        return f"""
다음은 사용자의 보험 비교 관련 질문입니다:
{query}

아래는 비교에 사용할 보험사별 정보입니다:
{context}

위 정보를 바탕으로 사용자 질문에 답변해주세요. 다음 지침을 따라주세요:
1. 보험사별 핵심 특징과 차이점을 명확하게 비교해주세요.
2. 보장 내용, 보험료, 특약, 면책 조항 등 중요한 요소를 비교해주세요.
3. 가능하면 비교 내용을 표 형식으로 정리해 보여주세요.
4. 각 보험사/상품의 장단점을 객관적으로 설명해주세요.
5. 알 수 없는 정보는 솔직히 모른다고 말해주세요.
6. 답변 마지막에는 참고한 보험사와 정보 출처를 명시해주세요.
"""
