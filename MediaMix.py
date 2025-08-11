# streamlit_app.py
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
import dropbox
from io import BytesIO

st.set_page_config(page_title="Reach 1+ Optimization")
st.title("📊 Reach 1+ Optimization")

APP_KEY        = st.secrets["dropbox"]["app_key"]
APP_SECRET     = st.secrets["dropbox"]["app_secret"]
REFRESH_TOKEN  = st.secrets["dropbox"]["refresh_token"]
DROPBOX_PATH   = st.secrets["dropbox"]["path"]

def load_from_dropbox(path, usecols=None, parse_dates=None):
    try:
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=REFRESH_TOKEN,
            app_key=APP_KEY,
            app_secret=APP_SECRET,
        )
        _, res = dbx.files_download(DROPBOX_PATH)
        content = res.content  # bytes
    except Exception as e:
        st.error(f"⚠ 데이터 불러오기 실패: {e}")
        return None
    try:
        return pd.read_csv(
            BytesIO(content),
            engine="pyarrow",
            usecols=usecols,
            parse_dates=parse_dates,
        )
    except Exception:
        try:
            return pd.read_csv(
                BytesIO(content),
                usecols=usecols,
                low_memory=False,
                encoding="utf-8-sig",
                parse_dates=parse_dates,
            )
        except Exception as e:
            st.error(f"⚠ CSV 파싱 실패: {e}")
            return None

needed_cols = ['date', 'brand_id', 'target', 'media', 'impression', 'r1']

df_raw = load_from_dropbox(
    path=DROPBOX_PATH,
    usecols=needed_cols,
    parse_dates=["date"]
)

# 로드 실패 시 중단
if df_raw is None:
    st.stop()

st.success("✅ 데이터 불러오기 성공")

# 0 제거
df = df_raw[df_raw['r1'] != 0].copy()

# Wide 변환 & 컬럼 매핑
pivot = df.pivot_table(
    index=['date', 'brand_id', 'target'],
    columns='media',
    values=['impression', 'r1'],
    aggfunc='sum'
)
pivot.columns = [f"{a}_{b}" for a, b in pivot.columns.to_flat_index()]
pivot = pivot.reset_index()

rename_map = {
    'impression_total': 'imps',
    'r1_total': 'r1',
    'impression_tv': 'imps_a',
    'r1_tv': 'r1_a',
    'impression_digital': 'imps_b',
    'r1_digital': 'r1_b',
}
for src, dst in rename_map.items():
    if src in pivot.columns:
        pivot.rename(columns={src: dst}, inplace=True)

required_cols = ['imps', 'r1', 'imps_a', 'r1_a', 'imps_b', 'r1_b']
missing = [c for c in required_cols if c not in pivot.columns]
if missing:
    st.error(f"필수 데이터가 없습니다: {missing}")
    st.stop()

df0 = pivot.dropna(subset=required_cols).copy()
df0['r1_ab'] = df0['r1_a'] * df0['r1_b']

# 타겟 선택
target_list = sorted(df0['target'].unique())
if not target_list:
    st.error("⚠️ 선택 가능한 타겟이 없습니다.")
    st.stop()

selected_target = st.selectbox("Target", target_list, index=0)
df_t = df0[df0['target'] == selected_target].reset_index(drop=True)

st.caption(f"✅ **{selected_target}** 데이터가 적용되었습니다.")

# 변수 바인딩
x     = df_t['imps'].values
r1    = df_t['r1'].values
x_a   = df_t['imps_a'].values
r1_a  = df_t['r1_a'].values
x_b   = df_t['imps_b'].values
r1_b  = df_t['r1_b'].values
r1_ab = df_t['r1_ab'].values

# 예측용 impressions (1부터 시작: 0분모 방지)
imps  = np.arange(1, 600_000_000, 1_000_000, dtype=np.int64)

# Hill 함수 & 피팅
def hill(x, a, b, c):
    return c / (1.0 + (b / x)**a)

initial_params = [1.0, 50_000_000.0, 0.6]
bounds_a = ([0, 0, 0], [np.inf, np.inf, 1.0])
bounds_b = ([0, 0, 0], [np.inf, np.inf, 0.7])

popt_a, _ = curve_fit(hill, x_a, r1_a, p0=initial_params, bounds=bounds_a, maxfev=30000)
popt_b, _ = curve_fit(hill, x_b, r1_b, p0=initial_params, bounds=bounds_b, maxfev=30000)
popt_t, _ = curve_fit(hill, x,   r1,   p0=initial_params, bounds=bounds_a, maxfev=30000)

pred_a    = hill(x_a, *popt_a)
pred_b    = hill(x_b, *popt_b)
pred_a_r1 = hill(imps, *popt_a)
pred_b_r1 = hill(imps, *popt_b)
pred_t_r1 = hill(imps, *popt_t)
pred_ab_r1= pred_a_r1 * pred_b_r1

# 성능표
media_r1_result = pd.DataFrame({
    'Hill n (a)': [popt_a[0], popt_b[0]],
    'EC50 (b)':   [popt_a[1], popt_b[1]],
    'Max (c)':    [popt_a[2], popt_b[2]],
    'R-squared':  [r2_score(r1_a, pred_a), r2_score(r1_b, pred_b)],
    'MAE(%)':     [mean_absolute_error(r1_a, pred_a)*100, mean_absolute_error(r1_b, pred_b)*100]
}, index=['TV','Digital'])

# 통합 Reach 1+
X = pd.DataFrame({
    'const': 0.0,
    'r1_a': df_t['r1_a'].values,
    'r1_b': df_t['r1_b'].values,
    'r1_ab': df_t['r1_ab'].values
})
model_total = sm.OLS(r1, X).fit()
coef_df_r1 = pd.DataFrame(model_total.params).T
coef_df_r1.index = ['Coefficient']
pred_r1 = model_total.predict(X)

# 출력/요약
##st.subheader("모델 요약")
##c1, c2, c3 = st.columns(3)
##with c1:
##    st.markdown("**미디어별 Reach 1+**")
##    st.dataframe(media_r1_result)
##with c2:
##    st.markdown("**통합 Reaach 1+**")
##    st.dataframe(coef_df_r1)
##with c3:
##    st.markdown("**모델 적합도**")
##    st.write(f"MSE: {mean_squared_error(r1, pred_r1) * 100:.3f}")
##    st.write(f"MAE: {mean_absolute_error(r1, pred_r1) * 100:.3f}")
##    st.write(f"R-squared: {r2_score(r1, pred_r1):.4f}")

# 시각화
st.subheader("미디어별 Reach 1+")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(imps, 100*pred_a_r1, alpha=0.6, s=10, label='TV', color='royalblue')
ax.scatter(imps, 100*pred_b_r1, alpha=0.6, s=10, label='Digital', color='darkorange')
ax.scatter(imps, 100*pred_t_r1, alpha=0.6, s=10, label='Total', color='mediumseagreen')

ax.set_xlabel("Impressions")
ax.set_ylabel("Reach 1+ (%)")
ax.grid(True, linestyle="--")
ax.legend()
st.pyplot(fig)

# 공통 함수
def analyze_custom_budgets(budget_a_eok, budget_b_eok, cpm_a, cpm_b):
    # 억 원 → 원
    budget_a = budget_a_eok * 100_000_000
    budget_b = budget_b_eok * 100_000_000

    # 각 매체별 imps
    imps_a = budget_a / (cpm_a / 1000.0) if cpm_a > 0 else 0.0
    imps_b = budget_b / (cpm_b / 1000.0) if cpm_b > 0 else 0.0

    pa = hill(np.array([imps_a]), *popt_a) if imps_a > 0 else np.array([0.0])
    pb = hill(np.array([imps_b]), *popt_b) if imps_b > 0 else np.array([0.0])
    pab = pa * pb

    X_custom = pd.DataFrame({
        'const': 0.0,
        'r1_a': pa,
        'r1_b': pb,
        'r1_ab': pab
    })
    pred_total = model_total.predict(X_custom)  # 0~1

    # 표
    df_custom = pd.DataFrame({
        '항목': ['TV(억 원)', 'Digital(억 원)', '총(억 원)', 'TV Reach1+(%)', 'Digital Reach1+(%)', '통합 Reach1+(%)'],
        '값': [
            np.round(budget_a/100_000_000, 2),
            np.round(budget_b/100_000_000, 2),
            np.round((budget_a+budget_b)/100_000_000, 2),
            np.round(100*pa[0], 2),
            np.round(100*pb[0], 2),
            np.round(100*pred_total[0], 2)
        ]
    })

    parts = {"pa": pa, "pb": pb, "pred_total": pred_total}
    return df_custom, parts

def optimize_mix_over_budget(cpm_a, cpm_b, max_budget_units=30, unit=100_000_000):
    results = []
    a = np.arange(0, 101, dtype=np.float64) / 100.0
    b = 1.0 - a
    for budget_range in range(1, max_budget_units + 1):
        budget = unit * budget_range
        imps_a = a * budget / (cpm_a / 1000.0)
        imps_b = b * budget / (cpm_b / 1000.0)
        pa = hill(imps_a, *popt_a)
        pb = hill(imps_b, *popt_b)
        pab = pa * pb
        X_mix = pd.DataFrame({'const': 0.0, 'r1_a': pa, 'r1_b': pb, 'r1_ab': pab})
        pred_i = model_total.predict(X_mix)
        optimal_idx = int(np.argmax(pred_i))
        results.append({
            '예산(억 원)': budget_range,
            'TV': f"{int(a[optimal_idx]*100)}%",
            'Digital': f"{int(b[optimal_idx]*100)}%",
            'Total Reach 1+(%)': round(100.0 * pred_i[optimal_idx], 2)
        })
    return results

def optimize_single_budget(budget_won, cpm_a, cpm_b, unit_points=100):
    a = np.arange(0, unit_points + 1) / 100.0
    b = 1.0 - a
    imps_a = a * budget_won / (cpm_a / 1000.0)
    imps_b = b * budget_won / (cpm_b / 1000.0)
    pa = hill(imps_a, *popt_a)
    pb = hill(imps_b, *popt_b)
    pab = pa * pb
    X_mix = pd.DataFrame({'const': 0.0, 'r1_a': pa, 'r1_b': pb, 'r1_ab': pab})
    pred_i = model_total.predict(X_mix)

    # 스플라인(부드러운 곡선)
    df_spline = pd.DataFrame({'a': a, 'pred': pred_i})
    spline_a = dmatrix("bs(a, df=12, degree=2, include_intercept=True)", data=df_spline, return_type='dataframe')
    spline_fit = sm.OLS(df_spline['pred'], spline_a).fit()
    spline_i = spline_fit.predict(spline_a)

    optimal_idx = int(np.argmax(pred_i))
    out = pd.DataFrame({
        'TV': [f"{int(a[optimal_idx]*100)}%"],
        'Digital': [f"{int(b[optimal_idx]*100)}%"],
        'Total Reach 1+(%)': [round(100.0 * pred_i[optimal_idx], 2)]
    })
    out = pd.DataFrame(out).reset_index(drop=True)
    return a, pred_i, spline_i, out

# UI: 탭
st.subheader("💰 예산 분석/최적화")

col_cpm1, col_cpm2 = st.columns(2)
with col_cpm1:
    cpm_a_global = st.number_input("CPM TV", value=9000, step=100, key="cpm_tv_global")
with col_cpm2:
    cpm_b_global = st.number_input("CPM Digital", value=7000, step=100, key="cpm_dg_global")
    
tab1, tab2, tab3 = st.tabs(["미디어별 예산 분석", "예산 범위 최적화", "특정 예산 최적화"])

# 세션 상태 (탭 이동해도 유지)
for key in ["custom_parts", "sweep_df", "single_curve", "single_out"]:
    if key not in st.session_state:
        st.session_state[key] = None

with tab1:
    c_budget_a, c_budget_b, c= st.columns([1, 1, 0.5])
    with c_budget_a:
        budget_a_eok = st.number_input("TV", value=3.5, step=0.1)
    with c_budget_b:
        budget_b_eok = st.number_input("Digital", value=3.5, step=0.1)
    with c:
        button1 = st.button("실행", type="primary", key="button1")

    if button1:
        st.session_state.custom_df, st.session_state.custom_parts = analyze_custom_budgets(
            budget_a_eok, budget_b_eok, cpm_a_global, cpm_b_global
        )

    if st.session_state.custom_df is not None:
        pa_val = float(st.session_state.custom_parts['pa'][0])
        pb_val = float(st.session_state.custom_parts['pb'][0])
        total_val = float(st.session_state.custom_parts['pred_total'][0])
        fig1, ax1 = plt.subplots(figsize=(6,4))
        bars = ax1.bar(['TV','Digital','Total'],
               [100*pa_val, 100*pb_val, 100*total_val],
               color=['royalblue', 'darkorange', 'mediumseagreen'])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                     f"{height:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylim(0, 100); ax1.set_ylabel("Reach 1+(%)")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig1)

with tab2:
    c1, c2 = st.columns([2, 0.5])
    with c1:
        max_units = st.slider("예산 범위(억 원)", min_value=1, max_value=30, value=15)
    with c2:
        button2 = st.button("실행", type="primary", key="button2")

    if button2:
        st.session_state.sweep_df = optimize_mix_over_budget(cpm_a_global, cpm_b_global, max_budget_units=max_units)

    if st.session_state.sweep_df is not None:
        st.dataframe(st.session_state.sweep_df, use_container_width=True)
        # 전액 A/B 비교선
        budgets = np.arange(1, max_units + 1) * 100_000_000
        imps_a_allA = budgets / (cpm_a_global / 1000.0)
        imps_b_allB = budgets / (cpm_b_global / 1000.0)
        pa_allA = hill(imps_a_allA, *popt_a)
        pb_allB = hill(imps_b_allB, *popt_b)

        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.plot(st.session_state.sweep_df['예산(억 원)'], st.session_state.sweep_df['Total Reach 1+(%)'], marker='o', label='Opt Mix', color='mediumseagreen')
        ax2.plot(np.arange(1, max_units+1), 100*pa_allA, linestyle='--', marker='s', label='Only TV', color='royalblue')
        ax2.plot(np.arange(1, max_units+1), 100*pb_allB, linestyle='--', marker='^', label='Only Digital', color='darkorange')
        ax2.set_xlabel("Budget Range"); ax2.set_ylabel("Reach 1+(%)")
        ax2.grid(True, linestyle='--'); ax2.legend()
        st.pyplot(fig2)

with tab3:
    c1, c2 = st.columns([2, 0.5])
    with c1:
        single_budget = st.number_input("특정 예산(억 원)", value=7.0, step=0.1)
    with c2:
        button3 = st.button("실행", type="primary", key="button3")

    if button3:
        a, pred_i, spline_i, out = optimize_single_budget(single_budget*100_000_000, cpm_a_global, cpm_b_global)
        st.session_state.single_curve = (a, pred_i, spline_i)
        st.session_state.single_out = out
        
    if st.session_state.single_out is not None:
        st.dataframe(st.session_state.single_out, use_container_width=True)

    if st.session_state.single_curve is not None:
        a, pred_i, spline_i = st.session_state.single_curve
        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.scatter(100*a, 100*pred_i, alpha=0.6, s=30, label='Predicted', color='gold')
        ax3.plot(100*a, 100*spline_i, color='crimson', linewidth=2, label='Spline Fit')
        ax3.set_xlabel('TV ratio (%)'); ax3.set_ylabel('Reach 1+ (%)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
