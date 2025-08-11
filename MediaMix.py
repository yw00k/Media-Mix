# streamlit_app.py
import streamlit as st
import time
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
    
msg_placeholder = st.empty()
msg_placeholder.success("✅ 데이터 불러오기 성공")
time.sleep(2)
msg_placeholder.empty()

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
imps  = np.arange(1, 200_000_000, 1_000_000, dtype=np.int64)

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
ax.set_ylabel("Reach 1+(%)")
ax.legend()
st.pyplot(fig)

# 공통 함수
def analyze_custom_budget(budget_a_eok, budget_b_eok, cpm_a, cpm_b):
    # 단위 환산
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
    pred_total = model_total.predict(X_custom)

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

    # 스플라인 추정
    df_spline = pd.DataFrame({'a': a, 'pred': pred_i})
    spline_a = dmatrix("bs(a, df=12, degree=2, include_intercept=True)", data=df_spline, return_type='dataframe')
    spline_fit = sm.OLS(df_spline['pred'], spline_a).fit()
    spline_i = spline_fit.predict(spline_a)

    optimal_idx = int(np.argmax(pred_i))
    out = pd.DataFrame({
        'TV': [f"{int(a[optimal_idx]*100)}%"],
        'Digital': [f"{int(b[optimal_idx]*100)}%"],
        'Total Reach 1+(%)': [round(100.0 * pred_i[optimal_idx], 2)]
    }).reset_index(drop=True)
    return a, pred_i, spline_i, out

def analyze_vs_opt(budget_a_eok, budget_b_eok, cpm_a, cpm_b, unit=100_000_000):
    # 단위 환산
    won_a = budget_a_eok * unit
    won_b = budget_b_eok * unit
    total_won = won_a + won_b
    total_eok = (won_a + won_b) / unit
    
    imps_a_user = won_a / (cpm_a / 1000.0) if cpm_a > 0 else 0.0
    imps_b_user = won_b / (cpm_b / 1000.0) if cpm_b > 0 else 0.0
    pa_user = hill(np.array([imps_a_user]), *popt_a)
    pb_user = hill(np.array([imps_b_user]), *popt_b)
    pab_user = pa_user * pb_user
    X_user = pd.DataFrame({'const': 0.0, 'r1_a': pa_user, 'r1_b': pb_user, 'r1_ab': pab_user})
    pred_user = float(model_total.predict(X_user)[0])

    a = np.arange(0, 101, dtype=np.float64) / 100.0
    b = 1.0 - a
    imps_a_opt = a * total_won / (cpm_a / 1000.0) if cpm_a > 0 else np.zeros_like(a)
    imps_b_opt = b * total_won / (cpm_b / 1000.0) if cpm_b > 0 else np.zeros_like(a)
    pa_opt_curve = hill(imps_a_opt, *popt_a)
    pb_opt_curve = hill(imps_b_opt, *popt_b)
    pab_opt_curve = pa_opt_curve * pb_opt_curve
    X_opt = pd.DataFrame({'const': 0.0, 'r1_a': pa_opt_curve, 'r1_b': pb_opt_curve, 'r1_ab': pab_opt_curve})
    pred_curve = model_total.predict(X_opt).values
    idx = int(np.argmax(pred_curve))

    a_opt, b_opt = float(a[idx]), float(b[idx])
    pa_opt, pb_opt = float(pa_opt_curve[idx]), float(pb_opt_curve[idx])
    pred_opt = float(pred_curve[idx])

    summary = pd.DataFrame([
        {
            '구분': '사용자안',
            'TV 예산(억)': round(won_a / unit, 2),
            'Digital 예산(억)': round(won_b / unit, 2),
            'TV 비중(%)': int(round(100 * won_a / total_won)) if total_won > 0 else 0,
            'Digital 비중(%)': int(round(100 * won_b / total_won)) if total_won > 0 else 0,
            'TV Reach1+(%)': round(100 * pa_user[0], 2),
            'Digital Reach1+(%)': round(100 * pb_user[0], 2),
            'Total Reach1+(%)': round(100 * pred_user, 2),
        },
        {
            '구분': '최적화안',
            'TV 예산(억)': round(total_eok * a_opt, 2),
            'Digital 예산(억)': round(total_eok * b_opt, 2),
            'TV 비중(%)': int(round(100 * a_opt)),
            'Digital 비중(%)': int(round(100 * b_opt)),
            'TV Reach1+(%)': round(100 * pa_opt, 2),
            'Digital Reach1+(%)': round(100 * pb_opt, 2),
            'Total Reach1+(%)': round(100 * pred_opt, 2),
        }
    ])

    return summary, pred_user, pred_opt
    
def optimize_mix_over_budget(cpm_a, cpm_b, max_budget_units=30, unit=100_000_000):
    a = np.arange(0, 101, dtype=np.float64) / 100.0
    b = 1.0 - a

    budget_eok = np.arange(1, max_budget_units + 1)
    budget_won = budget_eok * unit

    imps_o_a = budget_won / (cpm_a / 1000.0)
    imps_o_b = budget_won / (cpm_b / 1000.0)
    poa = hill(imps_o_a, *popt_a)
    pob = hill(imps_o_b, *popt_b)
    df_only = pd.DataFrame({
        '예산(억 원)': budget_eok,
        'Only TV': np.round(100 * poa, 2),
        'Only Digital': np.round(100 * pob, 2),
    }).reset_index(drop=True)

    results = []

    for won, eok in zip(budget_won, budget_eok):
        imps_a = a * won / (cpm_a / 1000.0)
        imps_b = b * won / (cpm_b / 1000.0)
        pa = hill(imps_a, *popt_a)
        pb = hill(imps_b, *popt_b)
        pab = pa * pb
        X_mix = pd.DataFrame({'const': 0.0, 'r1_a': pa, 'r1_b': pb, 'r1_ab': pab})
        pred_i = model_total.predict(X_mix)
        optimal_idx = int(np.argmax(pred_i))
        results.append({
            '예산(억 원)': eok,
            'TV': f"{int(a[optimal_idx]*100)}%",
            'Digital': f"{int(b[optimal_idx]*100)}%",
            'Total Reach 1+(%)': round(100.0 * pred_i[optimal_idx], 2)
        })

    df_opt = pd.DataFrame(results).reset_index(drop=True)

    return df_opt, df_only

# UI: 탭
st.subheader("💰 예산 분석/최적화")

col_cpm1, col_cpm2 = st.columns(2)
with col_cpm1:
    cpm_a_global = st.number_input("CPM TV", value=9000, step=100, key="cpm_tv_global")
with col_cpm2:
    cpm_b_global = st.number_input("CPM Digital", value=7000, step=100, key="cpm_dg_global")
    
tab1, tab2, tab3 = st.tabs(["입력 예산 분석", "입력 예산 최적화", "예산 범위 최적화"])

# 세션 상태 (탭 이동해도 유지)
for key in ["custom_parts", "single_curve", "single_out", "sweep_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

with tab1:
    c_budget_a, c_budget_b = st.columns([1, 1])
    with c_budget_a:
        budget_a_eok = st.number_input("TV", value=3.5, step=0.1)
    with c_budget_b:
        budget_b_eok = st.number_input("Digital", value=3.5, step=0.1)
    
    button1 = st.button("실행", type="primary", key="button1")

    if button1:
        summary_df, pred_user, pred_opt = analyze_vs_opt(
            budget_a_eok, budget_b_eok, cpm_a_global, cpm_b_global
        )
        st.session_state.user_vs_opt = (summary_df, pred_user, pred_opt)

    if st.session_state.get("user_vs_opt") is not None:
        summary_df, pred_user, pred_opt = st.session_state.user_vs_opt

        labels = ['TV', 'Digital', 'Total']

        user_vals = [
            summary_df.loc['TV Reach1+(%)', '사용자안'],
            summary_df.loc['Digital Reach1+(%)', '사용자안'],
            summary_df.loc['Total Reach1+(%)', '사용자안'],
        ]
        opt_vals = [
            summary_df.loc['TV Reach1+(%)', '최적화안'],
            summary_df.loc['Digital Reach1+(%)', '최적화안'],
            summary_df.loc['Total Reach1+(%)', '최적화안'],
        ]

        x = np.arange(len(labels))
        width = 0.38

        fig, ax = plt.subplots(figsize=(7, 4))
        bars1 = ax.bar(x - width/2, user_vals, width, label='사용자안', color='#6AADE4')
        bars2 = ax.bar(x + width/2, opt_vals,  width, label='최적화안', color='#43AA8B')

        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.2f}%", ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Reach 1+(%)")
        st.pyplot(fig)

        # 요약 테이블(1개)
        summary_wide = (summary_df.set_index('안').T)
        summary_wide = summary_wide[['사용자안', '최적화안']]
        st.dataframe(summary_wide, use_container_width=True)

with tab2:
    single_budget = st.number_input("특정 예산(억 원)", value=7.0, step=0.1)
    button2 = st.button("실행", type="primary", key="button2")

    if button2:
        a, pred_i, spline_i, out = optimize_single_budget(single_budget*100_000_000, cpm_a_global, cpm_b_global)
        st.session_state.single_curve = (a, pred_i, spline_i)
        st.session_state.single_out = out
        
    if st.session_state.single_out is not None:
        st.dataframe(st.session_state.single_out, use_container_width=True)

    if st.session_state.single_curve is not None:
        a, pred_i, spline_i = st.session_state.single_curve
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(100*a, 100*pred_i, alpha=0.6, s=30, label='Predicted', color='gold')
        ax2.plot(100*a, 100*spline_i, color='crimson', linewidth=2, label='Spline Fit')
        ax2.set_xlabel('TV ratio (%)'); ax2.set_ylabel('Reach 1+ (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

with tab3:
    max_units = st.slider("예산 범위(억 원)", min_value=1, max_value=30, value=15)
    button3 = st.button("실행", type="primary", key="button3")

    if button3:
        df_opt, df_only = optimize_mix_over_budget(cpm_a_global, cpm_b_global, max_budget_units=max_units)
        st.session_state.sweep_opt = df_opt
        st.session_state.sweep_only = df_only

    if (st.session_state.get("sweep_opt") is not None) and (st.session_state.get("sweep_only") is not None):
        df_opt  = st.session_state.sweep_opt
        df_only = st.session_state.sweep_only
        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.plot(df_opt['예산(억 원)'], df_opt['Total Reach 1+(%)'], marker='o', label='Opt Mix', color='mediumseagreen')
        ax3.plot(df_only['예산(억 원)'], df_only['Only TV'], linestyle='--', marker='s', label='Only TV', color='royalblue')
        ax3.plot(df_only['예산(억 원)'], df_only['Only Digital'], linestyle='--', marker='^', label='Only Digital', color='darkorange')
        ax3.set_xlabel("Budget Range"); ax3.set_ylabel("Reach 1+(%)")
        ax3.grid(True, linestyle='--'); ax3.legend()
        st.pyplot(fig3)

        st.dataframe(df_opt, use_container_width=True)
