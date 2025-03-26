"""
사용자 질문의 의도를 분류하는 모듈
"""
import os
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import openai


class IntentType(Enum):
    """질문 의도 유형"""
    TERMS_QUESTION = "약관_질문"  # 약관 관련 질문
    COMPARISON_QUESTION = "비교설계_질문"  # 보험사 비교 질문
    HYBRID_QUESTION = "복합_질문"  # 두 유형이 혼합된 질문
    IRRELEVANT_QUESTION = "무관련_질문"  # 보험과 무관한 질문
    FOLLOWUP_QUESTION = "후속_질문"  # 이전 대화 맥락의 후속 질문


class IntentClassifierService:
    """사용자 질문 의도 분류 서비스"""
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        IntentClassifierService 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            model: 사용할 모델명
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # 이전 대화 컨텍스트 저장
        self.conversation_history = []
        
    def classify_intent(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Tuple[IntentType, float, Dict[str, Any]]:
        """
        질문 의도 분류
        
        Args:
            question: 사용자 질문
            conversation_history: 이전 대화 이력 (선택)
            
        Returns:
            분류된 의도, 신뢰도 점수, 추가 메타 정보
        """
        if conversation_history:
            self.conversation_history = conversation_history
            
        # 대화 이력이 있으면 후속 질문인지 확인
        is_followup = len(self.conversation_history) > 0
        
        # 분류 프롬프트 구성
        prompt = self._create_classification_prompt(question, is_followup)
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": 
                     "너는 보험 질문을 정확하게 분류하는 전문가야. 질문을 분석하여 '약관_질문', '비교설계_질문', '복합_질문', '무관련_질문' 중 하나로 분류해야 해. "
                     "- 약관_질문: 특정 보험의 약관 내용, 보장 범위, 면책 사항 등에 관한 질문\n"
                     "- 비교설계_질문: 여러 보험사나 상품 간의 비교, 장단점, 추천 등에 관한 질문\n"
                     "- 복합_질문: 약관 내용과 상품 비교가 모두 포함된 질문\n"
                     "- 무관련_질문: 보험과 관련 없는 일반 질문이나 잡담\n"
                     "- 후속_질문: 이전 대화 맥락을 참조하는 질문\n"
                     "JSON 형식으로만 응답해. 형식: {\"intent\": \"의도유형\", \"confidence\": 0~1 사이 신뢰도, \"meta\": {\"키워드\": [관련키워드], \"mentioned_companies\": [언급된보험사], \"is_comparison\": true/false}}"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            # 응답 파싱
            result = response.choices[0].message.content
            
            try:
                import json
                parsed = json.loads(result)
                
                intent_str = parsed.get("intent", "무관련_질문")
                confidence = float(parsed.get("confidence", 0.5))
                meta = parsed.get("meta", {})
                
                # 문자열을 IntentType으로 변환
                intent = None
                for intent_type in IntentType:
                    if intent_type.value == intent_str:
                        intent = intent_type
                        break
                
                # 일치하는 IntentType이 없으면 기본값 사용
                if intent is None:
                    intent = IntentType.IRRELEVANT_QUESTION
                
                # 대화 이력 업데이트
                self.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                
                return intent, confidence, meta
                
            except Exception as e:
                print(f"응답 파싱 중 오류 발생: {str(e)}")
                return IntentType.IRRELEVANT_QUESTION, 0.0, {}
            
        except Exception as e:
            print(f"의도 분류 중 오류 발생: {str(e)}")
            return IntentType.IRRELEVANT_QUESTION, 0.0, {}
    
    def _create_classification_prompt(self, question: str, is_followup: bool) -> str:
        """
        분류 프롬프트 생성
        
        Args:
            question: 사용자 질문
            is_followup: 후속 질문 여부
            
        Returns:
            분류 프롬프트
        """
        prompt = f"다음 질문을 '약관_질문', '비교설계_질문', '복합_질문', '무관련_질문', '후속_질문' 중 하나로 분류해주세요:\n\n{question}\n\n"
        
        if is_followup:
            prompt += "이전 대화 맥락:\n"
            for msg in self.conversation_history[-3:]:  # 최근 3개 메시지만 사용
                role = "사용자" if msg["role"] == "user" else "시스템"
                prompt += f"[{role}] {msg['content']}\n"
        
        prompt += "\n분석 과정:\n1. 질문이 약관 내용, 보장 범위, 면책사항 등에 관한 것인지 확인\n2. 질문이 여러 보험사나 상품 비교에 관한 것인지 확인\n3. 두 유형이 모두 포함되어 있는지 확인\n4. 보험과 무관한 질문인지 확인\n5. 이전 대화 맥락을 참조하는 후속 질문인지 확인\n\nJSON 형식으로 응답해주세요."
        
        return prompt
    
    def update_conversation_history(self, role: str, content: str):
        """
        대화 이력 업데이트
        
        Args:
            role: 메시지 역할 ('user' 또는 'assistant')
            content: 메시지 내용
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # 대화 이력 길이 제한 (최근 10개 메시지만 유지)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        현재 대화 이력 반환
        
        Returns:
            대화 이력 목록
        """
        return self.conversation_history.copy()
