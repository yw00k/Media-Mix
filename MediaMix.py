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
        content = res.content
    except Exception as e:
        st.error(f"⚠ 데이터 불러오기 실패: {e}")
        return None
    try:
        return pd.read_csv(BytesIO(content), engine="pyarrow", usecols=usecols, parse_dates=parse_dates)
    except Exception:
        try:
            return pd.read_csv(BytesIO(content), usecols=usecols, low_memory=False, encoding="utf-8-sig", parse_dates=parse_dates)
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
x_total = df_t['imps'].values
y_total = df_t['r1'].values
x_a     = df_t['imps_a'].values   # a = TV
y_a     = df_t['r1_a'].values
x_b     = df_t['imps_b'].values   # b = Digital
y_b     = df_t['r1_b'].values
y_ab    = df_t['r1_ab'].values

# 예측용 impressions (1부터 시작: 0분모 방지)
imps  = np.arange(1, 200_000_000, 1_000_000, dtype=np.int64)

# Hill 함수 & 피팅
def hill(x, a, b, c):
    return c / (1.0 + (b / x)**a)

initial_params = [1.0, 50_000_000.0, 0.6]
bounds_a = ([0, 0, 0], [np.inf, np.inf, 1.0])
bounds_b = ([0, 0, 0], [np.inf, np.inf, 0.7])

popt_a, _ = curve_fit(hill, x_a, y_a, p0=initial_params, bounds=bounds_a, maxfev=30000)
popt_b, _ = curve_fit(hill, x_b, y_b, p0=initial_params, bounds=bounds_b, maxfev=30000)
popt_t, _ = curve_fit(hill, x_total, y_total, p0=initial_params, bounds=bounds_a, maxfev=30000)

pred_a_fit = hill(x_a, *popt_a)
pred_b_fit = hill(x_b, *popt_b)

pred_a_curve = hill(imps, *popt_a)
pred_b_curve = hill(imps, *popt_b)
pred_t_curve = hill(imps, *popt_t)

# 성능표 (내부 네이밍은 a/b/total, 출력 라벨은 TV/Digital/Total)
media_r1_result = pd.DataFrame({
    'Hill n (a)': [popt_a[0], popt_b[0]],
    'EC50 (b)':   [popt_a[1], popt_b[1]],
    'Max (c)':    [popt_a[2], popt_b[2]],
    'R-squared':  [r2_score(y_a, pred_a_fit), r2_score(y_b, pred_b_fit)],
    'MAE(%)':     [mean_absolute_error(y_a, pred_a_fit)*100, mean_absolute_error(y_b, pred_b_fit)*100]
}, index=['TV','Digital'])

# 통합 모델 학습
X_train = pd.DataFrame({
    'const': 0.0,
    'r1_a': df_t['r1_a'].values,
    'r1_b': df_t['r1_b'].values,
    'r1_ab': df_t['r1_ab'].values
})
model_total = sm.OLS(y_total, X_train).fit()
pred_in = model_total.predict(X_train)

# 시각화: 미디어별 Reach 1+(%)
st.subheader("미디어별 Reach 1+(%)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(imps, 100*pred_a_curve, alpha=0.6, s=10, label='TV', color='royalblue')
ax.scatter(imps, 100*pred_b_curve, alpha=0.6, s=10, label='Digital', color='darkorange')
ax.scatter(imps, 100*pred_t_curve, alpha=0.6, s=10, label='Total', color='mediumseagreen')
ax.set_xlabel("Impressions")
ax.set_ylabel("Reach 1+(%)")
ax.legend()
st.pyplot(fig)

# 공통 계산 함수들
def analyze_custom_budget(a_eok, b_eok, cpm_a, cpm_b, unit=100_000_000):
    # 억→원
    a_won = a_eok * unit
    b_won = b_eok * unit

    # 각 매체별 imps
    a_imps = a_won / (cpm_a / 1000.0) if cpm_a > 0 else 0.0
    b_imps = b_won / (cpm_b / 1000.0) if cpm_b > 0 else 0.0

    a_r1 = hill(np.array([a_imps]), *popt_a) if a_imps > 0 else np.array([0.0])
    b_r1 = hill(np.array([b_imps]), *popt_b) if b_imps > 0 else np.array([0.0])
    ab_r1 = a_r1 * b_r1

    X_user = pd.DataFrame({
        'const': 0.0,
        'r1_a': a_r1,
        'r1_b': b_r1,
        'r1_ab': ab_r1
    })
    total_r1 = model_total.predict(X_user)

    df_out = pd.DataFrame({
        '항목': ['TV(억 원)', 'Digital(억 원)', '총(억 원)', 'TV Reach 1+(%)', 'Digital Reach 1+(%)', 'Total Reach 1+(%)'],
        '값': [
            np.round(a_won/unit, 2),
            np.round(b_won/unit, 2),
            np.round((a_won+b_won)/unit, 2),
            np.round(100*a_r1[0], 2),
            np.round(100*b_r1[0], 2),
            np.round(100*total_r1[0], 2)
        ]
    })
    parts = {'a_r1': a_r1, 'b_r1': b_r1, 'total_r1': total_r1}
    return df_out, parts

def optimize_total_budget(a_eok, b_eok, cpm_a, cpm_b, unit=100_000_000):
    total_won = (a_eok + b_eok) * unit
    a = np.arange(0, 101, dtype=np.float64) / 100.0
    b = 1.0 - a

    a_imps = a * total_won / (cpm_a / 1000.0) if cpm_a > 0 else np.zeros_like(a)
    b_imps = b * total_won / (cpm_b / 1000.0) if cpm_b > 0 else np.zeros_like(a)

    a_r1_curve = hill(a_imps, *popt_a)
    b_r1_curve = hill(b_imps, *popt_b)
    ab_r1_curve = a_r1_curve * b_r1_curve

    X_opt = pd.DataFrame({
        'const': 0.0,
        'r1_a': a_r1_curve,
        'r1_b': b_r1_curve,
        'r1_ab': ab_r1_curve
    })
    total_r1_curve = model_total.predict(X_opt).values
    idx = int(np.argmax(total_r1_curve))

    out = {
        'a_share': float(a[idx]),
        'b_share': float(b[idx]),
        'a_r1': float(a_r1_curve[idx]),
        'b_r1': float(b_r1_curve[idx]),
        'total_r1': float(total_r1_curve[idx])
    }
    return out

def compare_user_vs_opt(a_eok, b_eok, cpm_a, cpm_b, unit=100_000_000):
    # 사용자안
    user_df, user_parts = analyze_custom_budget(a_eok, b_eok, cpm_a, cpm_b, unit)
    user_a_r1 = float(user_parts['a_r1'][0])
    user_b_r1 = float(user_parts['b_r1'][0])
    user_total_r1 = float(user_parts['total_r1'][0])

    # 최적화안 (총액 고정)
    opt = optimize_total_budget(a_eok, b_eok, cpm_a, cpm_b, unit)
    total_eok = a_eok + b_eok
    a_eok_opt = round(total_eok * opt['a_share'], 2)
    b_eok_opt = round(total_eok * opt['b_share'], 2)

    summary = pd.DataFrame([
        {
            '구분': '사용자안',
            'TV 예산(억)': round(a_eok, 2),
            'Digital 예산(억)': round(b_eok, 2),
            'TV 비중': f"{int(round(100 * (a_eok / total_eok))) if total_eok>0 else 0}%",
            'Digital 비중': f"{int(round(100 * (b_eok / total_eok))) if total_eok>0 else 0}%",
            'TV Reach 1+(%)': round(100 * user_a_r1, 2),
            'Digital Reach 1+(%)': round(100 * user_b_r1, 2),
            'Total Reach 1+(%)': round(100 * user_total_r1, 2),
        },
        {
            '구분': '최적화안',
            'TV 예산(억)': a_eok_opt,
            'Digital 예산(억)': b_eok_opt,
            'TV 비중': f"{int(round(100 * opt['a_share']))}%",
            'Digital 비중': f"{int(round(100 * opt['b_share']))}%",
            'TV Reach 1+(%)': round(100 * opt['a_r1'], 2),
            'Digital Reach 1+(%)': round(100 * opt['b_r1'], 2),
            'Total Reach 1+(%)': round(100 * opt['total_r1'], 2),
        }
    ])
    return summary

# 예산 범위 최적화
def optimize_mix_over_budget(cpm_a, cpm_b, max_budget_units=30, unit=100_000_000):
    a = np.arange(0, 101, dtype=np.float64) / 100.0
    b = 1.0 - a

    budget_eok = np.arange(1, max_budget_units + 1)
    budget_won = budget_eok * unit

    # Only TV/Digital 라인 (참고/플롯용)
    a_imps_only = budget_won / (cpm_a / 1000.0)
    b_imps_only = budget_won / (cpm_b / 1000.0)
    only_a = hill(a_imps_only, *popt_a)
    only_b = hill(b_imps_only, *popt_b)
    df_only = pd.DataFrame({
        '예산(억 원)': budget_eok,
        'Only TV': np.round(100 * only_a, 2),
        'Only Digital': np.round(100 * only_b, 2),
    }).reset_index(drop=True)

    results = []
    for won, eok in zip(budget_won, budget_eok):
        a_imps = a * won / (cpm_a / 1000.0)
        b_imps = b * won / (cpm_b / 1000.0)
        a_r1 = hill(a_imps, *popt_a)
        b_r1 = hill(b_imps, *popt_b)
        ab_r1 = a_r1 * b_r1
        X_mix = pd.DataFrame({'const': 0.0, 'r1_a': a_r1, 'r1_b': b_r1, 'r1_ab': ab_r1})
        total_r1_curve = model_total.predict(X_mix)
        idx = int(np.argmax(total_r1_curve))
        results.append({
            '예산(억 원)': eok,
            'TV 비중': f"{int(a[idx]*100)}%",
            'Digital 비중': f"{int(b[idx]*100)}%",
            'Total Reach 1+(%)': round(100.0 * total_r1_curve[idx], 2)
        })

    df_opt = pd.DataFrame(results).reset_index(drop=True)
    return df_opt, df_only

# UI
st.subheader("💰 예산 최적화")

col_cpm1, col_cpm2 = st.columns(2)
with col_cpm1:
    cpm_a_global = st.number_input("CPM TV", value=9000, step=100, key="cpm_tv_global")
with col_cpm2:
    cpm_b_global = st.number_input("CPM Digital", value=7000, step=100, key="cpm_dg_global")

tab1, tab2, tab3 = st.tabs(["개별 예산 최적화", "총 예산 최적화", "예산 범위 최적화"])

# 세션 상태 (접두어 통일: compare_/single_/sweep_)
for key in ["compare_result", "single_curve", "single_out", "sweep_opt", "sweep_only"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 탭1: 사용자안 vs 최적화안 비교
with tab1:
    c_a, c_b = st.columns([1, 1])
    with c_a:
        a_eok_input = st.number_input("TV 예산(억 원)", value=3.5, step=0.1)
    with c_b:
        b_eok_input = st.number_input("Digital 예산(억 원)", value=3.5, step=0.1)

    if st.button("실행", type="primary", key="compare_run"):
        summary_df = compare_user_vs_opt(a_eok_input, b_eok_input, cpm_a_global, cpm_b_global)
        st.session_state.compare_result = summary_df

    if st.session_state.compare_result is not None:
        summary_df = st.session_state.compare_result

        summary_wide = summary_df.set_index('구분').T.rename_axis('항목')
        summary_wide = summary_wide[['사용자안', '최적화안']]

        # 막대 비교(각 안별 TV/Digital/Total Reach 1+(%))
        labels = ['TV', 'Digital', 'Total']
        user_vals = [
            summary_wide.loc['TV Reach 1+(%)', '사용자안'],
            summary_wide.loc['Digital Reach 1+(%)', '사용자안'],
            summary_wide.loc['Total Reach 1+(%)', '사용자안'],
        ]
        opt_vals = [
            summary_wide.loc['TV Reach 1+(%)', '최적화안'],
            summary_wide.loc['Digital Reach 1+(%)', '최적화안'],
            summary_wide.loc['Total Reach 1+(%)', '최적화안'],
        ]

        x = np.arange(len(labels))
        width = 0.38
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bars1 = ax1.bar(x - width/2, user_vals, width, label='User', color='yellow')
        bars2 = ax1.bar(x + width/2, opt_vals,  width, label='Opt', color='olivedrab')
        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                ax1.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.2f}%", ha='center', va='bottom', fontsize=9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Reach 1+(%)")
        ax1.legend()
        st.pyplot(fig1)

        st.dataframe(summary_wide, use_container_width=True)

# 탭2: 총 예산 최적 비중 곡선
with tab2:
    total_eok_input = st.number_input("총 예산(억 원)", value=7.0, step=0.1)
    if st.button("실행", type="primary", key="single_run"):

        a = np.arange(0, 101, dtype=np.float64) / 100.0
        b = 1.0 - a
        won = total_eok_input * 100_000_000
        a_imps = a * won / (cpm_a_global / 1000.0)
        b_imps = b * won / (cpm_b_global / 1000.0)
        a_r1 = hill(a_imps, *popt_a)
        b_r1 = hill(b_imps, *popt_b)
        ab_r1 = a_r1 * b_r1
        X_mix = pd.DataFrame({'const': 1.0, 'r1_a': a_r1, 'r1_b': b_r1, 'r1_ab': ab_r1})
        pred = model_total.predict(X_mix)

        df_spline = pd.DataFrame({'a': a, 'pred': pred})
        spline_a = dmatrix("bs(a, df=12, degree=2, include_intercept=True)", data=df_spline, return_type='dataframe')
        spline_fit = sm.OLS(df_spline['pred'], spline_a).fit()
        spline_i = spline_fit.predict(spline_a)

        st.session_state.single_curve = (a, pred.values, spline_i)
        best_idx = int(np.argmax(pred))
        out = pd.DataFrame({
            'TV 비중': [f"{int(a[best_idx]*100)}%"],
            'Digital 비중': [f"{int(b[best_idx]*100)}%"],
            'Total Reach 1+(%)': [round(100.0 * float(pred[best_idx]), 2)]
        })
        st.session_state.single_out = out

    if st.session_state.single_curve is not None:
        a, pred_i, spline_i = st.session_state.single_curve
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(100*a, 100*pred_i, alpha=0.6, s=30, label='Predicted', color='gold')
        ax2.plot(100*a, 100*spline_i, color='crimson', linewidth=2, label='Spline Fit')
        ax2.set_xlabel('TV ratio (%)')
        ax2.set_ylabel('Reach 1+(%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)
    if st.session_state.single_out is not None:
        st.dataframe(st.session_state.single_out, use_container_width=True)

# 탭3: 예산 범위 최적화(스윕)
with tab3:
    max_units = st.slider("예산 범위(억 원)", min_value=1, max_value=30, value=15)
    if st.button("실행", type="primary", key="sweep_run"):
        df_opt, df_only = optimize_mix_over_budget(cpm_a_global, cpm_b_global, max_budget_units=max_units)
        st.session_state.sweep_opt = df_opt
        st.session_state.sweep_only = df_only

    if (st.session_state.sweep_opt is not None) and (st.session_state.sweep_only is not None):
        df_opt  = st.session_state.sweep_opt
        df_only = st.session_state.sweep_only
        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.plot(df_opt['예산(억 원)'], df_opt['Total Reach 1+(%)'], marker='o', label='Opt Mix', color='mediumseagreen')
        ax3.plot(df_only['예산(억 원)'], df_only['Only TV'], linestyle='--', marker='s', label='Only TV', color='royalblue')
        ax3.plot(df_only['예산(억 원)'], df_only['Only Digital'], linestyle='--', marker='^', label='Only Digital', color='darkorange')
        ax3.set_xlabel("Budget Range")
        ax3.set_ylabel("Reach 1+(%)")
        ax3.grid(True, linestyle='--'); ax3.legend()
        st.pyplot(fig3)

        st.dataframe(df_opt, use_container_width=True)
