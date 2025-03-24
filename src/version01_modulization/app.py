import uvicorn
import config
import argparse
from models import get_embedding_model
from rag_service import RAGService

def main():
    """애플리케이션 메인 엔트리 포인트"""
    parser = argparse.ArgumentParser(description='InsuPanda API 서버')
    
    # 명령행 인자 추가
    parser.add_argument('--host', type=str, default='0.0.0.0', help='호스트 주소')
    parser.add_argument('--port', type=int, default=8000, help='포트 번호')
    parser.add_argument('--embedding-model', type=str, default='upstage', help='임베딩 모델 유형 (upstage, mock 등)')
    parser.add_argument('--llm-model', type=str, default=config.DEFAULT_LLM_MODEL, help='LLM 모델 이름')
    parser.add_argument('--reload', action='store_true', help='개발 모드에서 코드 변경 시 자동 리로드')
    
    args = parser.parse_args()
    
    # 임베딩 모델 초기화 테스트
    print(f"사용하는 임베딩 모델: {args.embedding_model}")
    try:
        # 테스트 텍스트로 임베딩 모델 예열
        embedding_model = get_embedding_model(args.embedding_model)
        print("임베딩 모델 초기화 성공!")
    except Exception as e:
        print(f"임베딩 모델 초기화 실패: {e}")
        print("기본 임베딩 모델로 계속합니다.")
    
    print(f"LLM 모델: {args.llm_model}")
    
    # 서버 시작
    print(f"서버 시작: http://{args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 