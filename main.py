import re
import os
import openai
import mysql.connector
import json
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
from config import DEFAULT_CONFIG, DB_CONFIG
from prompt import BASE_PROMPT
from schema import DB_SCHEMA

load_dotenv()


def make_temporary_jsonfile(temp_data):
    if not os.path.isdir("./query_results"):
        os.makedirs("query_results", exist_ok=True)

    # 원본 결과를 임시 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"query_results/temp_result_{timestamp}.json"

    with open(temp_filename, "w", encoding="utf-8") as f:
        json.dump(temp_data, f, ensure_ascii=False, indent=2, default=str)

    print("\n임시 파일이 생성되었습니다:", temp_filename)


def execute_sql_query(generated_sql: str, used_config: dict) -> Optional[list]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # 쿼리 실행
        cursor.execute(generated_sql)
        results = cursor.fetchall()

        print("\n[검색 결과]")
        if results:
            print(f"전체 결과 수: {len(results)}개")

            # 결과를 그대로 저장 (JSON 변환 없이)
            temp_data = {
                "설정값": used_config,
                "쿼리": generated_sql,
                "결과": [
                    dict(row) for row in results
                ],  # SQL 결과를 그대로 딕셔너리로 변환
            }

            make_temporary_jsonfile(temp_data)

            # 결과 출력 (처음 20개만)
            for idx, row in enumerate(results[:10], 1):
                print(f"\n결과 {idx}:")
                for key, value in row.items():
                    print(f"{key}: {value}")

            # print("\n2_json_converter.py를 실행하여 최종 JSON을 생성하세요.")

        else:
            print("검색 결과가 없습니다.")

        cursor.close()
        conn.close()
        return results

    except Exception as e:
        print(f"\n❌ SQL 실행 오류: {str(e)}")
        return None


def generate_sql_query(
    prompt: str, age: int, sex: int, product_type: str, expiry_year: str
) -> str:
    try:
        client = openai.OpenAI()
        system_prompt = BASE_PROMPT.format(
            schema=DB_SCHEMA,
            age=age,
            sex_num=sex,
            sex="남자" if sex == 1 else "여자",
            product_type=product_type,
            expiry_year=expiry_year,
        )

        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # SQL 쿼리 추출 및 정제
        sql_query = response.choices[0].message.content.strip()

        # 마크다운 코드 블록 제거
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        return sql_query
    except Exception as e:
        print(f"\n❌ GPT API 오류: {str(e)}")
        return None


def process_query(prompt: str):
    # 현재 사용되는 설정값 저장
    current_config = DEFAULT_CONFIG.copy()
    current_config["expiry_year"] = (
        f"{DEFAULT_CONFIG['expiry']}y_{DEFAULT_CONFIG['duration']}"
    )

    # 보험사 추출 (옵션)
    if "삼성" in prompt:
        current_config["company_id"] = "01"
    elif "한화" in prompt:
        current_config["company_id"] = "02"

    # 다른 보험사들에 대한 매핑도 추가 가능
    generated_sql = generate_sql_query(
        prompt=prompt,
        age=current_config["insu_age"],
        sex=current_config["sex"],
        product_type=current_config["product_type"],
        expiry_year=current_config["expiry_year"],
    )

    # 설정값과 쿼리 출력
    print("\n=== 실행 결과 ===")
    print("\n[설정값]")
    print(f"이름: {current_config['custom_name']}")
    print(f"나이: {current_config['insu_age']}세")
    print(f"성별: {'남자' if current_config['sex'] == 1 else '여자'}")
    print(
        f"상품유형: {'무해지형' if current_config['product_type'] == 'nr' else '해지환급형'}"
    )
    print(f"보험기간: {current_config['expiry_year']}")
    print(
        f"보험사ID: {current_config['company_id'] if current_config['company_id'] else '지정되지 않음'}"
    )

    if generated_sql:
        print("\n[실행 쿼리]")
        print(generated_sql)
        return execute_sql_query(generated_sql, current_config)
    return print("❌ SQL문이 생성되지 않았습니다.")


if __name__ == "__main__":
    # try:
    #     user_name = input("이름 :").strip()
    #     user_age = input("나이 :")
    #     user_sex = input("성별 여자 0, 남자 1 입력")
    #     user_type = input("무해지형: nr, 해지환급형: r 입력")
    #     user_expiry = input("만기 몇년? :")
    #     user_duration = input("보장기간 몇세?:")
    # except Exception as e:
    #     print(f"예상치 못한 오류가 발생했습니다: {e}")

    # DEFAULT_CONFIG["custom_name"] = user_name
    # DEFAULT_CONFIG["insu_age"] = user_age
    # DEFAULT_CONFIG["sex"] = user_sex
    # DEFAULT_CONFIG["product_type"] = user_type
    # DEFAULT_CONFIG["expiry"] = user_expiry
    # DEFAULT_CONFIG["duration"] = user_duration

    print("\n=== 보험 상담 챗봇 ===")

    user_question = input(
        "질문을 입력하세요 (종료하려면 'q' 또는 'quit' 입력):\n"
    ).strip()
    process_query(user_question)
