"""
보험사 매핑 정보를 관리하는 모듈
"""


class InsuranceMappings:
    """보험사 이름, 별칭, 컬렉션 간의 매핑을 관리하는 클래스"""

    def __init__(self):
        """보험사 관련 매핑 정보 초기화"""
        # 키워드에서 컬렉션 이름으로의 매핑
        self.keyword_to_collection = {
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

        # 컬렉션 이름에서 실제 보험사 이름으로의 역매핑
        self.collection_to_company = {
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

        # 보험사별 키워드 목록
        self.company_keywords = {
            "삼성화재": [
                "삼성화재",
                "삼성",
                "samsung",
            ],
            "DB손해보험": [
                "DB손해보험",
                "db손해보험",
                "DB손보",
                "db손보",
                "디비손해보험",
                "디비손보",
                "디비",
            ],
            "하나손해보험": ["하나손해보험", "하나손보", "하나", "hana"],
            "한화손해보험": ["한화손해보험", "한화손보", "한화", "hanwha"],
            "흥국화재": ["흥국화재", "흥국", "heung"],
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
        self.insurance_types = [
            "실손",
            "암보험",
            "정기보험",
            "종신보험",
            "연금보험",
            "저축보험",
            "어린이보험",
            "치아보험",
            "운전자보험",
            "여행자보험",
            "화재보험",
            "자동차보험",
            "통합보험",
            "건강보험",
            "상해보험",
            "재물보험",
        ]

    def get_collection_name(self, keyword):
        """키워드에 해당하는 컬렉션 이름 반환"""
        return self.keyword_to_collection.get(keyword)

    def get_company_name(self, collection_name):
        """컬렉션 이름에 해당하는 보험사 이름 반환"""
        return self.collection_to_company.get(collection_name, collection_name)

    def detect_companies_in_question(self, question):
        """사용자 질문에서 언급된 보험사 목록 추출"""
        mentioned_companies = []
        
        # 각 보험사별로 키워드 확인
        for company, keywords in self.company_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    mentioned_companies.append(company)
                    break
        
        return list(set(mentioned_companies))  # 중복 제거

    def is_comparison_question(self, question):
        """비교 질문인지 확인"""
        comparison_keywords = ["비교", "차이", "다른점", "같은점", "어떤것", "어느것", "뭐가", "어디가", "vs", "대", "어떤게"]
        
        # 비교 키워드가 있는지 확인
        has_comparison_keyword = any(keyword in question for keyword in comparison_keywords)
        
        # 언급된 보험사 수 확인
        mentioned_companies = self.detect_companies_in_question(question)
        multiple_companies = len(mentioned_companies) >= 2
        
        return has_comparison_keyword or multiple_companies
