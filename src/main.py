import openai
import mysql.connector
from dotenv import load_dotenv
from config import DEFAULT_CONFIG, DB_CONFIG
from prompts import BASE_PROMPT, EXAMPLE_PROMPT, INTENT_PROMPT
from schema import DB_SCHEMA
from utils import process_query

load_dotenv()
client = openai.OpenAI()


class IntentModule:
    def __init__(self, user_question: str, PROMPT):
        self.prompt = PROMPT.format(question=user_question)

    def classify_response(self):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "너는 GA 보험설계사들이 사용하는 보험전문 챗봇이야.",
                },
                {"role": "user", "content": self.prompt},
            ],
        )
        return response.choices[0].message.content


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


class CompareModule:
    def __init__(self):
        self.default_config = DEFAULT_CONFIG

    def setting_information(self, config):
        # 설정값과 쿼리 출력
        print("\n=== 실행 결과 ===")
        print("\n[설정값]")
        print(f"이름: {config['custom_name']}")
        print(f"나이: {config['insu_age']}세")
        print(f"성별: {'남자' if config['sex'] == 1 else '여자'}")
        print(
            f"상품유형: {'무해지형' if config['product_type'] == 'nr' else '해지환급형'}"
        )
        print(f"보험기간: {config['expiry_year']}")

    def handle_prompt(self, user_question: str) -> str:
        prompt, current_config = process_query(user_question, self.default_config)
        generated_sql = generate_sql_query(prompt, current_config)
        self.setting_information(current_config)
        result = execute_sql_query(generated_sql, current_config)
        return print(result)


if __name__ == "__main__":

    print("\n=== 보험 상담 챗봇 ===")

    user_question = input(
        "질문을 입력하세요 (종료하려면 'q' 또는 'quit' 입력):\n"
    ).strip()

    classify_intent = IntentModule(user_question, INTENT_PROMPT)
    compare_module = CompareModule()
    result_intent = classify_intent.classify_response()

    if result_intent == "비교설계 질문":
        print(result_intent)
        compare_module.handle_prompt(user_question)
    else:
        print("version 01 붙일 예정")
