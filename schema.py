# DB 스키마 정의
DB_SCHEMA = """-- 1. 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS insu DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE insu; 

-- 2. 보험 비교 설계 테이블 (comparison)
CREATE TABLE comparison (
    custom_name VARCHAR(255) NOT NULL COMMENT '성명',
    insu_age TINYINT UNSIGNED NOT NULL COMMENT '보험나이',
    sex TINYINT(1) NOT NULL COMMENT '성별 (0: 여자, 1: 남자)',
    product_type VARCHAR(50) NOT NULL COMMENT '상품유형',
    expiry_year VARCHAR(10) NOT NULL COMMENT '만기',
    company_id VARCHAR(20) NOT NULL COMMENT '보험사ID',
    product_id VARCHAR(20) NOT NULL COMMENT '보험ID',
    coverage_id VARCHAR(20) NOT NULL COMMENT '보장항목ID',
    premium_amount INT NOT NULL COMMENT '보험료',
    PRIMARY KEY (insu_age, sex, product_type, company_id, product_id, coverage_id)
);

-- 3. 보장항목 테이블 (coverage)
CREATE TABLE coverage (
    coverage_id VARCHAR(20) NOT NULL COMMENT '보장항목ID',
    coverage_name VARCHAR(255) NOT NULL COMMENT '보장항목명',
    default_coverage_amount DECIMAL(15,2) NOT NULL COMMENT '초기세팅 보험금',
    is_default TINYINT(1) NOT NULL COMMENT '1: default, 0: not default',
    PRIMARY KEY (coverage_id)
);

-- 4. 보험사 테이블 (insu_company)
CREATE TABLE insu_company (
    company_id VARCHAR(20) NOT NULL COMMENT '보험사ID',
    company_name VARCHAR(255) NOT NULL COMMENT '보험사명',
    is_default TINYINT(1) NOT NULL COMMENT '1: default, 0: not default',
    PRIMARY KEY (company_id)
);

-- 5. 보험상품 테이블 (insu_product)
CREATE TABLE insu_product (
    company_id VARCHAR(20) NOT NULL COMMENT '보험사ID',
    product_id VARCHAR(20) NOT NULL COMMENT '보험ID',
    product_name VARCHAR(255) NOT NULL COMMENT '보험명',
    is_default TINYINT(1) NOT NULL COMMENT '1: default, 0: not default',
    PRIMARY KEY (company_id, product_id),
    FOREIGN KEY (company_id) REFERENCES insu_company(company_id) ON DELETE CASCADE
);"""
