# InsuPanda: 보험 약관 검색 및 질의응답 서비스

InsuPanda는 보험 약관 데이터를 벡터화하여 저장하고, 사용자 질문에 대해 관련 정보를 검색하여 답변을 생성하는 RAG(Retrieval-Augmented Generation) 기반 서비스입니다.

## 주요 기능

- 보험 약관 텍스트 기반 벡터 검색
- 질문에 맞는 컬렉션 자동 선택
- 여러 보험사 약관 비교 분석
- 검색 결과 기반 자연어 답변 생성

## 시스템 구조

본 프로젝트는 다음과 같은 모듈로 구성되어 있습니다:

1. **임베딩 모듈(`models.py`)**: 다양한 임베딩 모델을 추상화하고 통합 인터페이스 제공
2. **벡터 저장소(`vector_store.py`)**: FAISS 기반 벡터 인덱스 관리 및 검색 기능
3. **RAG 서비스(`rag_service.py`)**: 검색 및 답변 생성 로직 구현
4. **API 서버(`server.py`)**: FastAPI 기반 웹 API 제공
5. **설정 관리(`config.py`)**: 환경 변수 및 기본 설정 관리
6. **애플리케이션(`app.py`)**: 메인 애플리케이션 엔트리 포인트

## 설치 및 실행 방법

### 사전 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)

### 설치

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/insupanda.git
cd insupanda
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 항목을 설정하세요:
```
UPSTAGE_API_KEY=your_upstage_api_key
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key  # 선택 사항
LANGCHAIN_PROJECT=insupanda             # 선택 사항
```

### 실행

기본 설정으로 서버 시작:
```bash
python app.py
```

다른 임베딩 모델이나 포트로 실행:
```bash
python app.py --embedding-model mock --port 9000
```

사용 가능한 명령행 옵션:
```
--host HOST           호스트 주소 (기본값: 0.0.0.0)
--port PORT           포트 번호 (기본값: 8000)
--embedding-model MODEL  임베딩 모델 유형 (기본값: upstage)
--llm-model MODEL     LLM 모델 이름 (기본값: gpt-4o-mini)
--reload              개발 모드에서 코드 변경 시 자동 리로드
```

## API 사용 방법

### POST /api/search

JSON 형식으로 쿼리 및 컬렉션 지정:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "삼성화재 암보험에 대해 알려줘", "collections": ["Samsung_YakMu2404103NapHae20250113"]}'
```

### GET /api/search

URL 쿼리 파라미터로 쿼리 및 컬렉션 지정:

```bash
curl "http://localhost:8000/api/search?query=삼성화재%20암보험에%20대해%20알려줘&collections=Samsung_YakMu2404103NapHae20250113"
```

## 임베딩 모델 확장

새로운 임베딩 모델을 추가하려면 `models.py`에 `EmbeddingModel` 추상 클래스를 상속하는 새 클래스를 정의하고, `get_embedding_model()` 함수에 해당 모델을 추가하세요:

```python
class NewEmbeddingModel(EmbeddingModel):
    def __init__(self, **kwargs):
        # 초기화 코드
        pass
        
    def get_embedding(self, text: str) -> np.ndarray:
        # 임베딩 생성 코드
        pass

# get_embedding_model 함수에 새 모델 추가
def get_embedding_model(model_type: str = "upstage", **kwargs) -> EmbeddingModel:
    if model_type.lower() == "upstage":
        return UpstageEmbeddingModel(**kwargs)
    elif model_type.lower() == "mock":
        return MockEmbeddingModel(**kwargs)
    elif model_type.lower() == "new":  # 새 모델 추가
        return NewEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"지원되지 않는 임베딩 모델 유형: {model_type}")
```

## 라이센스

이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다.
