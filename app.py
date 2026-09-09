import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# ----------------------------------------------------
# 1. 기본 설정 및 한글 폰트 적용
# ----------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# OS별 한글 폰트 설정 (깨짐 방지)
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)

# ----------------------------------------------------
# 2. 데이터 로드 및 전처리
# ----------------------------------------------------
@st.cache_data
def load_data():
    # 데이터 로드
    baci_df = pd.read_csv("baci_85_sample.csv")
    country_df = pd.read_csv("country_codes_sample.csv")
    
    # 결측치 요약 정보 저장
    missing_summary = baci_df.isnull().sum().to_frame(name="결측치 수")
    
    # 국가 코드 매핑 (수출국 i 기준)
    # 컬럼명에 따라 매칭 (i/country_code/iso 등 일반적인 포맷 반영)
    c_code_col = 'country_code' if 'country_code' in country_df.columns else country_df.columns[0]
    c_name_col = 'country_name' if 'country_name' in country_df.columns else country_df.columns[1]
    
    mapping_dict = dict(zip(country_df[c_code_col], country_df[c_name_col]))
    
    # BACI 데이터 컬럼 소문자 변환
    baci_df.columns = [c.lower() for c in baci_df.columns]
    
    # 수출국명 매핑 (i: 수출국 코드)
    baci_df['exporter_name'] = baci_df['i'].map(mapping_dict).fillna(baci_df['i'].astype(str))
    
    # 무역액 등급 (v: 거래액 기준 3분위수 대/중/소 구분)
    # v 컬럼 결측/이상치 대비
    baci_df['v'] = pd.to_numeric(baci_df['v'], errors='coerce').fillna(0)
    labels = ['소', '중', '대']
    baci_df['trade_grade'] = pd.qcut(baci_df['v'], q=3, labels=labels, duplicates='drop')
    
    return baci_df, missing_summary

try:
    df, missing_df = load_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오지 못했습니다. 파일 위치와 파일명을 확인해주세요: {e}")
    st.stop()

# ----------------------------------------------------
# 3. 사이드바 필터
# ----------------------------------------------------
st.sidebar.header("🔍 검색 필터")

# 국가 선택
all_countries = sorted(df['exporter_name'].unique().tolist())
selected_countries = st.sidebar.multiselect("국가 선택", options=all_countries, default=all_countries[:10])

# 무역액 등급 선택 (대, 중, 소)
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
missing_t = missing_df.T
st.dataframe(missing_t, use_container_width=True)

st.markdown("---")

# 3. 총 거래건수 & 총 수출액(달러)
st.subheader("📊 주요 통계 지표")
col1, col2 = st.columns(2)

total_deals = len(filtered_df)
total_export_value = filtered_df['v'].sum() * 1000 # BACI v단위는 일반적으로 1,000 USD

with col1:
    st.metric(label="총 거래건수", value=f"{total_deals:,} 건")
with col2:
    st.metric(label="총 수출액 (달러)", value=f"${total_export_value:,.2f}")

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급분포
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔥 국가*연도 수출액 히트맵 (상위 8개국)")
    # 상위 8개국 추출
    top8_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(8).index
    heatmap_data = filtered_df[filtered_df['exporter_name'].isin(top8_countries)]
    
    if not heatmap_data.empty:
        # t: 연도, exporter_name: 국가, v: 무역액
        pivot_heat = heatmap_data.pivot_table(index='exporter_name', columns='t', values='v', aggfunc='sum', fill_value=0)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar=True)
        ax.set_ylabel("국가명")
        ax.set_xlabel("연도")
        st.pyplot(fig)
    else:
        st.info("표시할 데이터가 없습니다.")

with col4:
    st.subheader("📦 무역액 등급분포")
    if not filtered_df.empty:
        grade_counts = filtered_df['trade_grade'].value_counts().reindex(['소', '중', '대']).fillna(0)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#85c1e9', '#f8c471', '#e74c3c']
        ax.bar(grade_counts.index, grade_counts.values, color=colors)
        ax.set_xlabel("무역액 등급")
        ax.set_ylabel("건수")
        
        # 바 위에 건수 표시
        for i, v in enumerate(grade_counts.values):
            ax.text(i, v, f"{int(v):,}", ha='center', va='bottom')
            
        st.pyplot(fig)
    else:
        st.info("표시할 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
st.subheader("📑 상위 5개국 × 무역액 등급 교차표")

top5_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(5).index
cross_data = filtered_df[filtered_df['exporter_name'].isin(top5_countries)]

if not cross_data.empty:
    col5, col6 = st.columns(2)
    
    # 원본 건수 교차표
    ct_counts = pd.crosstab(
        cross_data['exporter_name'], 
        cross_data['trade_grade'], 
        margins=True, 
        margins_name="합계"
    )
    
    # 정규화 비율 교차표 (행 기준 백분율)
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
    st.info("선택된 조건에 해당하는 상위 국가 데이터가 없습니다.")