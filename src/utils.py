import re
import mysql.connector
import openai
import simplejson as json
from schema import DB_SCHEMA
from config import DEFAULT_CONFIG, DB_CONFIG
from prompts import BASE_PROMPT, EXAMPLE_PROMPT, INTENT_PROMPT

client = openai.OpenAI()


def process_query(prompt: str, config):
    # 현재 사용되는 설정값 저장
    current_config = config.copy()

    # 나이 추출 (숫자 + "세" 패턴)
    age_match = re.search(r"(\d+)세", prompt)
    if age_match:
        current_config["insu_age"] = int(age_match.group(1))

    # 성별 추출
    if "남성" in prompt or "남자" in prompt:
        current_config["sex"] = 1
    elif "여성" in prompt or "여자" in prompt:
        current_config["sex"] = 0

    # 상품유형 추출
    if "무해지" in prompt:
        current_config["product_type"] = "nr"
    elif "해지환급" in prompt:
        current_config["product_type"] = "r"

    # 보험기간 추출
    period_match = re.search(r"(\d+)년[/\s](\d+)세", prompt)
    if period_match:
        years = period_match.group(1)
        age = period_match.group(2)
        current_config["expiry_year"] = f"{years}y_{age}"

    # 보험사 추출 (옵션)
    if "삼성" in prompt:
        current_config["company_id"] = "01"
    elif "한화" in prompt:
        current_config["company_id"] = "02"
    # 다른 보험사들에 대한 매핑도 추가 가능

    return prompt, current_config


def convert_sql_to_json_format(generate_json_data, EXAMPLE_PROMPT) -> str:

    converter_json_data = json.dumps(
        generate_json_data, ensure_ascii=False, indent=2, use_decimal=True
    )

    prompt = EXAMPLE_PROMPT + converter_json_data
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": "당신은 JSON 데이터 변환 전문가입니다. 주어진 예시 형식에 맞게 데이터를 변환해주세요.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    response_text = response.choices[0].message.content

    json_str = response_text.replace("```json", "").replace("```", "").strip()
    return json_str


def execute_sql_query(generated_sql: str, used_config: dict) -> str:
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
            "결과": [dict(row) for row in results],  # SQL 결과를 그대로 딕셔너리로 변환
        }

        json_result = convert_sql_to_json_format(temp_data, EXAMPLE_PROMPT)
        return json_result
    else:
        print("검색 결과가 없습니다.")

    cursor.close()
    conn.close()
    return results


def generate_sql_query(prompt, config) -> str:
    system_prompt = BASE_PROMPT.format(
        schema=DB_SCHEMA,
        age=config["insu_age"],
        sex_num=config["sex"],
        sex="남자" if config["sex"] == 1 else "여자",
        product_type=config["product_type"],
        expiry_year=config["expiry_year"],
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
