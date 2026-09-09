import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from pathlib import Path

# ----------------------------------------------------
# 1. 기본 설정 및 한글 폰트 적용
# ----------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# OS별 한글 폰트 설정 (깨짐 방지)
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')
else:  # Linux (Streamlit Cloud 환경 포함)
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)

# ----------------------------------------------------
# 2. 데이터 로드 및 전처리
# ----------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    
    # 파일 탐색 (루트 또는 dummy 폴더 내 위치 대응)
    baci_file = next((p for p in [base_dir / "baci_85_sample.csv", base_dir / "dummy" / "baci_85_sample.csv"] if p.exists()), None)
    country_file = next((p for p in [base_dir / "country_codes_sample.csv", base_dir / "dummy" / "country_codes_sample.csv"] if p.exists()), None)

    if not baci_file or not country_file:
        raise FileNotFoundError("CSV 데이터 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")

    baci_df = pd.read_csv(baci_file)
    country_df = pd.read_csv(country_file)

    # 결측치 요약 정보 저장
    missing_summary = baci_df.isnull().sum().to_frame(name="결측치 수")

    # 컬럼명 소문자 통일 및 공백 제거
    baci_df.columns = [c.lower().strip() for c in baci_df.columns]
    country_df.columns = [c.lower().strip() for c in country_df.columns]

    # 국가 코드 테이블 컬럼 탐색
    c_code_col = next((c for c in country_df.columns if any(k in c for k in ['country_code', 'code', 'id', 'iso', 'c_code'])), country_df.columns[0])
    c_name_col = next((c for c in country_df.columns if any(k in c for k in ['country_name', 'name', 'country', 'c_name'])), country_df.columns[1])

    # 정수형(int)으로 정규화하여 딕셔너리 매핑
    country_df[c_code_col] = pd.to_numeric(country_df[c_code_col], errors='coerce')
    country_df = country_df.dropna(subset=[c_code_col])
    mapping_dict = dict(zip(country_df[c_code_col].astype(int), country_df[c_name_col].astype(str)))

    # BACI 데이터: 한국(410) 단일 수출 데이터셋 대비
    # i(수출국)의 고유값이 1개이고 j(수입국/상대국)가 다양할 경우 j를 기준으로 매핑
    target_country_col = 'i'
    if 'j' in baci_df.columns and baci_df['i'].nunique() == 1:
        target_country_col = 'j'

    baci_df['country_num'] = pd.to_numeric(baci_df[target_country_col], errors='coerce')
    baci_df['exporter_name'] = baci_df['country_num'].map(mapping_dict).fillna("미식별(" + baci_df[target_country_col].astype(str) + ")")

    # 무역액 수치 변환
    baci_df['v'] = pd.to_numeric(baci_df['v'], errors='coerce').fillna(0)

    # 무역액 등급 (거래액 기준 3분위수 대/중/소 구분)
    try:
        labels = ['소', '중', '대']
        baci_df['trade_grade'] = pd.qcut(baci_df['v'], q=3, labels=labels, duplicates='drop')
    except Exception:
        # 데이터가 너무 적거나 중복값이 많을 경우 pd.cut으로 폴백
        baci_df['trade_grade'] = pd.cut(baci_df['v'], bins=3, labels=['소', '중', '대'])

    return baci_df, missing_summary

try:
    df, missing_df = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# ----------------------------------------------------
# 3. 사이드바 필터
# ----------------------------------------------------
st.sidebar.header("🔍 검색 필터")

all_countries = sorted(df['exporter_name'].dropna().unique().tolist())
default_selection = all_countries[:min(10, len(all_countries))]
selected_countries = st.sidebar.multiselect("국가 선택", options=all_countries, default=default_selection)

grades = ['대', '중', '소']
selected_grades = st.sidebar.multiselect("무역액 등급 선택", options=grades, default=grades)

# 필터링 적용
filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['exporter_name'].isin(selected_countries)]
if selected_grades:
    filtered_df = filtered_df[filtered_df['trade_grade'].isin(selected_grades)]

# ----------------------------------------------------
# 4. 메인 화면 출력
# ----------------------------------------------------

# 1. 타이틀
st.title("무역 분석 대시보드")
st.markdown("---")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("📋 baci_85_sample.csv 결측치 현황")
st.dataframe(missing_df.T, use_container_width=True)

st.markdown("---")

# 3. 총 거래건수 & 총 수출액(달러)
st.subheader("📊 주요 통계 지표")
col1, col2 = st.columns(2)

total_deals = len(filtered_df)
total_export_value = filtered_df['v'].sum() * 1000  # BACI 단위: 천 달러 -> 달러 변환

with col1:
    st.metric(label="총 거래건수", value=f"{total_deals:,} 건")
with col2:
    st.metric(label="총 수출액 (달러)", value=f"${total_export_value:,.2f}")

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급분포
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔥 국가 × 연도 수출액 히트맵 (상위 8개국)")
    top8_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(8).index
    heatmap_data = filtered_df[filtered_df['exporter_name'].isin(top8_countries)]
    
    if not heatmap_data.empty and 't' in heatmap_data.columns:
        pivot_heat = heatmap_data.pivot_table(index='exporter_name', columns='t', values='v', aggfunc='sum', fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar=True)
        ax.set_ylabel("국가명")
        ax.set_xlabel("연도")
        st.pyplot(fig)
    else:
        st.info("표시할 조건의 데이터가 없습니다.")

with col4:
    st.subheader("📦 무역액 등급분포")
    if not filtered_df.empty:
        grade_counts = filtered_df['trade_grade'].value_counts().reindex(['소', '중', '대']).fillna(0)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#85c1e9', '#f8c471', '#e74c3c']
        ax.bar(grade_counts.index.astype(str), grade_counts.values, color=colors)
        ax.set_xlabel("무역액 등급")
        ax.set_ylabel("건수")
        
        for i, v in enumerate(grade_counts.values):
            ax.text(i, v, f"{int(v):,}", ha='center', va='bottom')
            
        st.pyplot(fig)
    else:
        st.info("표시할 조건의 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
st.subheader("📑 상위 5개국 × 무역액 등급 교차표")

top5_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(5).index
cross_data = filtered_df[filtered_df['exporter_name'].isin(top5_countries)]

if not cross_data.empty:
    col5, col6 = st.columns(2)
    
    # 원본 건수
    ct_counts = pd.crosstab(
        cross_data['exporter_name'], 
        cross_data['trade_grade'], 
        margins=True, 
        margins_name="합계"
    )
    
    # 정규화 비율 (행 기준 백분율)
    ct_norm = pd.crosstab(
        cross_data['exporter_name'], 
        cross_data['trade_grade'], 
        normalize='index'
    ) * 100
    
    with col5:
        st.markdown("**1) 원본 건수 (건)**")
        st.dataframe(ct_counts, use_container_width=True)
        
    with col6:
        st.markdown("**2) 정규화 비율 (%)**")
        st.dataframe(ct_norm.style.format("{:.2f}%"), use_container_width=True)
else:
    st.info("선택된 조건에 해당하는 데이터가 없습니다.")