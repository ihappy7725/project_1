import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ----------------------------------------------------
# 1. 기본 설정 및 Fade-in 애니메이션 적용
# ----------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", page_icon="🚢", layout="wide")

# 부드럽게 떠오르는 Fade-in CSS 주입
st.markdown("""
<style>
@keyframes fadeIn {
    from { 
        opacity: 0; 
        transform: translateY(20px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}
/* Streamlit 메인 컨테이너에 애니메이션 적용 */
.block-container {
    animation: fadeIn 1.2s ease-out forwards;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. 데이터 로드 및 전처리
# ----------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    
    baci_file = next((p for p in [base_dir / "baci_85_sample.csv", base_dir / "dummy" / "baci_85_sample.csv"] if p.exists()), None)
    country_file = next((p for p in [base_dir / "country_codes_sample.csv", base_dir / "dummy" / "country_codes_sample.csv"] if p.exists()), None)

    if not baci_file or not country_file:
        raise FileNotFoundError("CSV 파일을 찾을 수 없습니다. baci_85_sample.csv 및 country_codes_sample.csv 파일 위치를 확인하세요.")

    baci_df = pd.read_csv(baci_file)
    country_df = pd.read_csv(country_file)

    missing_summary = baci_df.isnull().sum().to_frame(name="결측치 수")

    baci_df.columns = [c.lower().strip() for c in baci_df.columns]
    country_df.columns = [c.lower().strip() for c in country_df.columns]

    c_code_col = next((c for c in country_df.columns if any(k in c for k in ['country_code', 'code', 'id', 'iso', 'c_code'])), country_df.columns[0])
    c_name_col = next((c for c in country_df.columns if any(k in c for k in ['country_name', 'name', 'country', 'c_name'])), country_df.columns[1])

    country_df[c_code_col] = pd.to_numeric(country_df[c_code_col], errors='coerce')
    country_df = country_df.dropna(subset=[c_code_col])
    mapping_dict = dict(zip(country_df[c_code_col].astype(int), country_df[c_name_col].astype(str)))

    target_country_col = 'j' if ('j' in baci_df.columns and baci_df['i'].nunique() == 1) else 'i'

    baci_df['country_num'] = pd.to_numeric(baci_df[target_country_col], errors='coerce')
    baci_df['exporter_name'] = baci_df['country_num'].map(mapping_dict).fillna("미식별(" + baci_df[target_country_col].astype(str) + ")")

    baci_df['v'] = pd.to_numeric(baci_df['v'], errors='coerce').fillna(0)

    try:
        baci_df['trade_grade'] = pd.qcut(baci_df['v'], q=3, labels=['소', '중', '대'], duplicates='drop')
    except Exception:
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
with st.sidebar:
    st.title("⚙️ 분석 필터")
    st.markdown("분석할 조건을 선택하세요.")
    
    all_countries = sorted(df['exporter_name'].dropna().unique().tolist())
    default_selection = all_countries[:min(10, len(all_countries))]
    selected_countries = st.multiselect("📍 국가 선택", options=all_countries, default=default_selection)
    
    selected_grades = st.multiselect("📈 무역액 등급", options=['대', '중', '소'], default=['대', '중', '소'])

filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['exporter_name'].isin(selected_countries)]
if selected_grades:
    filtered_df = filtered_df[filtered_df['trade_grade'].isin(selected_grades)]

# ----------------------------------------------------
# 4. 메인 대시보드 화면
# ----------------------------------------------------
st.title("🚢 글로벌 무역 분석 대시보드")
st.markdown("선택된 국가와 무역액 등급에 따른 수출입 흐름과 분포를 분석합니다.")

with st.expander("📋 데이터 결측치 현황 확인하기 (baci_85_sample.csv)"):
    st.dataframe(missing_df.T, use_container_width=True)

st.divider()

# 주요 통계 지표 
total_deals = len(filtered_df)
total_export_value = filtered_df['v'].sum() * 1000

col1, col2 = st.columns(2)
with col1:
    st.metric(label="📦 총 거래건수", value=f"{total_deals:,} 건", delta="전체 기간 합산")
with col2:
    st.metric(label="💰 총 무역액 (USD)", value=f"${total_export_value:,.0f}", delta="단위: 달러")

st.divider()

# 시각화 영역 (Plotly 적용)
col3, col4 = st.columns([1.2, 1])

with col3:
    st.subheader("🌍 국가 × 연도 무역액 히트맵")
    top8_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(8).index
    heatmap_data = filtered_df[filtered_df['exporter_name'].isin(top8_countries)]
    
    if not heatmap_data.empty and 't' in heatmap_data.columns:
        pivot_heat = heatmap_data.pivot_table(index='exporter_name', columns='t', values='v', aggfunc='sum', fill_value=0)
        
        fig_heat = px.imshow(
            pivot_heat, 
            color_continuous_scale='Blues',
            aspect="auto",
            labels=dict(x="연도", y="국가명", color="무역액(USD)")
        )
        
        fig_heat.update_traces(
            hovertemplate="<b>국가명:</b> %{y}<br><b>연도:</b> %{x}<br><b>무역액:</b> $%{z:,.0f}<extra></extra>"
        )
        fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("조건에 맞는 데이터가 부족합니다.")

with col4:
    st.subheader("📊 무역액 등급분포")
    if not filtered_df.empty:
        grade_counts = filtered_df['trade_grade'].value_counts().reindex(['대', '중', '소']).fillna(0)
        
        blue_palette = ['#08519c', '#3182bd', '#9ecae1'] 
        
        fig_pie = px.pie(
            names=grade_counts.index,
            values=grade_counts.values,
            hole=0.45,
            color_discrete_sequence=blue_palette
        )
        
        fig_pie.update_traces(
            textinfo='none', 
            hovertemplate="<b>등급:</b> %{label}<br><b>건수:</b> %{value:,.0f}건<br><b>비율:</b> %{percent}<extra></extra>"
        )
        
        fig_pie.add_annotation(
            text=f"<b>총 {int(grade_counts.sum()):,}건</b>", 
            showarrow=False, 
            font=dict(size=16)
        )
        
        fig_pie.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5) 
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("조건에 맞는 데이터가 부족합니다.")

st.divider()

# 교차표 영역
st.subheader("🤝 상위 5개국 × 무역액 등급 교차표")
top5_countries = filtered_df.groupby('exporter_name')['v'].sum().nlargest(5).index
cross_data = filtered_df[filtered_df['exporter_name'].isin(top5_countries)]

if not cross_data.empty:
    col5, col6 = st.columns(2)
    
    ct_counts = pd.crosstab(cross_data['exporter_name'], cross_data['trade_grade'], margins=True, margins_name="총계")
    ct_norm = pd.crosstab(cross_data['exporter_name'], cross_data['trade_grade'], normalize='index') * 100
    
    with col5:
        st.markdown("**1. 원본 거래건수** (단위: 건)")
        st.dataframe(ct_counts.style.background_gradient(cmap='Blues', axis=None), use_container_width=True)
        
    with col6:
        st.markdown("**2. 등급별 비율** (단위: %)")
        st.dataframe(ct_norm.style.format("{:.1f}%").background_gradient(cmap='Blues', axis=None), use_container_width=True)
else:
    st.info("데이터가 없습니다.")