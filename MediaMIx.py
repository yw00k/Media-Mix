import streamlit as st
import dropbox
import pandas as pd
from io import BytesIO

# Streamlit secrets에서 정보 가져오기
APP_KEY = st.secrets["dropbox"]["app_key"]
APP_SECRET = st.secrets["dropbox"]["app_secret"]
REFRESH_TOKEN = st.secrets["dropbox"]["refresh_token"]
DROPBOX_PATH = st.secrets["dropbox"]["path"]

# Dropbox 객체 생성
dbx = dropbox.Dropbox(
    oauth2_refresh_token=REFRESH_TOKEN,
    app_key=APP_KEY,
    app_secret=APP_SECRET
)

def load_from_dropbox():
    try:
        _, res = dbx.files_download(DROPBOX_PATH)
        return pd.read_excel(BytesIO(res.content))
    except dropbox.exceptions.ApiError as e:
        st.error(f"⚠ API 오류: {e}")
        return None    
    except Exception as e:
        st.error(f"⚠ 서버에서 파일을 찾을 수 없습니다: {e}")
        return None

df_raw = load_from_dropbox()


# 데이터 로드
df_raw = load_from_dropbox()

if df_raw is not None:
    st.success("✅ 서버에서 데이터 불러오기 성공")
    st.dataframe(df_raw.head())

import importlib
import subprocess
import sys

# 필요한 패키지 목록
required_packages = [
    "streamlit",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "numpy",
    "pandas",
    "matplotlib",
    "patsy",
    "dropbox",
    "openpyxl"
]

# 패키지 설치
def install_and_import(package):
    try:
        importlib.import_module(package if package != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"📦 {package} 패키지가 설치되어 있지 않아 설치합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package if package != "scikit-learn" else "sklearn")

# 모든 패키지 확인 & 설치
for pkg in required_packages:
    install_and_import(pkg)

#Reach1.py
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix

st.set_page_config()
st.title("📊 미디어 믹스")

# 데이터 전처리

# 필요한 6개 컬럼 확인
required_cols_src = ['date', 'brand_id', 'target', 'media', 'impression', 'r1']
missing_src = [c for c in required_cols_src if c not in df_raw.columns]
if missing_src:
    st.error("⚠ 필수 데이터가 없습니다: {missing_src}")
    st.stop()

# reach < 0.01 제거
df = df_raw[df_raw['r1'] >= 0.01].copy()

# wide 변환
pivot = df.pivot_table(
    index=['date', 'brand_id', 'target'],
    columns='media',
    values=['impression', 'r1'],
    aggfunc='sum'
)
pivot.columns = [f"{a}_{b}" for a, b in pivot.columns.to_flat_index()]
pivot = pivot.reset_index()

# 컬럼 매핑
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
    st.error("⚠ 데이터가 없습니다.: {missing}")
    st.stop()

df0 = pivot.dropna(subset=required_cols).copy()
df0['r1_ab'] = df0['r1_a'] * df0['r1_b']

target_list = sorted(df0['target'].unique())
selected_target = st.selectbox("Target", target_list)

df = df0[df0['target'] == selected_target].reset_index(drop=True)
cpm_a_input = st.number_input("TV CPM", value=9137, step=100)
cpm_b_input = st.number_input("Digital CPM", value=7297, step=100)
st.write(f"✅ {selected_target} 데이터가 적용되었습니다.")

# 변수 바인딩
x, r1 = df['imps'].values, df['r1'].values
x_a, r1_a = df['imps_a'].values, df['r1_a'].values
x_b, r1_b = df['imps_b'].values, df['r1_b'].values

df_pred = pd.DataFrame.from_dict({
    'imps': np.arange(1, 800_000_001, 1_000_000, dtype=np.int64),
    'total_imps': np.arange(1, 800_000_001, 2_000_000, dtype=np.int64)
}, orient='index').T
imps = df_pred['imps'].values
total_imps = df_pred['total_imps'].values

def hill(x, a, b, c):
    return c / (1.0 + (b / x)**a)

# 미디어별 Reach 1+ 추정
initial_params = [1.0, 50_000_000.0, 0.6]
bounds_a = ([0, 0, 0], [np.inf, np.inf, 1.0])
bounds_b = ([0, 0, 0], [np.inf, np.inf, 0.7])

popt_a, _ = curve_fit(hill, x_a, r1_a, p0=initial_params, bounds=bounds_a, maxfev=20000)
popt_b, _ = curve_fit(hill, x_b, r1_b, p0=initial_params, bounds=bounds_b, maxfev=20000)

pred_a = hill(x_a, *popt_a)
pred_b = hill(x_b, *popt_b)

pred_a_r1 = hill(imps, *popt_a)
pred_b_r1 = hill(imps, *popt_b)
pred_ab_r1 = pred_a_r1 * pred_b_r1

media_r1_result = pd.DataFrame({
    'Hill n (a)': [popt_a[0], popt_b[0]],
    'EC50 (b)': [popt_a[1], popt_b[1]],
    'Max (c)': [popt_a[2], popt_b[2]],
    'R-squared': [r2_score(r1_a, pred_a), r2_score(r1_b, pred_b)],
    'MAE(%)': [mean_absolute_error(r1_a, pred_a) * 100,
               mean_absolute_error(r1_b, pred_b) * 100]
}, index=['Media A', 'Media B'])

# 통합 Reach 1+ 추정
X = pd.DataFrame({
    'const': 0.0,
    'r1_a': df['r1_a'].values,
    'r1_b': df['r1_b'].values,
    'r1_ab': df['r1_ab'].values
})
model_total = sm.OLS(r1, X).fit()
coef_df_r1 = pd.DataFrame(model_total.params).T
coef_df_r1.index = ['Coefficient']
pred_r1 = model_total.predict(X)

X_1 = pd.DataFrame({
    'const': 0.0,
    'r1_a': pred_a_r1,
    'r1_b': pred_b_r1,
    'r1_ab': pred_ab_r1
})
pred_total_r1 = model_total.predict(X_1)

st.subheader("Reach 1+")

# 시각화
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 4))
axes[0].scatter(imps, 100 * pred_a_r1, alpha=0.6, color='royalblue', s=30)
axes[0].set_title("TV")
axes[0].grid(True, linestyle='--')
axes[0].set_xlabel('TV Impressions')
axes[0].set_ylabel('Reach 1+ (%)')

axes[1].scatter(imps, 100 * pred_b_r1, alpha=0.6, color='darkorange', s=30)
axes[1].set_title("Digital")
axes[1].grid(True, linestyle='--')
axes[1].set_xlabel('Digital Impressions')

axes[2].scatter(total_imps, 100 * pred_total_r1, alpha=0.6, color='mediumseagreen', s=30)
axes[2].set_title("Total")
axes[2].grid(True, linestyle='--')
axes[2].set_xlabel('Total Impressions')
st.pyplot(fig, use_container_width=False)

# 예산 범위 최적화
def optimize_mix_over_budget(cpm_a, cpm_b, max_budget_units=30, unit=100_000_000):
    results = []
    a = np.arange(1, 100) / 100.0
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
            'TV 비율': f"{int(a[optimal_idx]*100)}%",
            'Digital 비율': f"{int(b[optimal_idx]*100)}%",
            '총 IMPs(억)': round((imps_a[optimal_idx] + imps_b[optimal_idx]) / 100_000_000, 2),
            'Reach 1+(%)': round(100.0 * pred_i[optimal_idx], 2)
        })
    return pd.DataFrame(results)

# 특정 예산 최적화
def optimize_single_budget(budget_won, cpm_a, cpm_b, unit_points=99):
    a = np.arange(1, unit_points + 1) / 100.0
    b = 1.0 - a
    imps_a = a * budget_won / (cpm_a / 1000.0)
    imps_b = b * budget_won / (cpm_b / 1000.0)
    pa = hill(imps_a, *popt_a)
    pb = hill(imps_b, *popt_b)
    pab = pa * pb
    X_mix = pd.DataFrame({'const': 0.0, 'r1_a': pa, 'r1_b': pb, 'r1_ab': pab})
    pred_i = model_total.predict(X_mix)
    spline_a = dmatrix("bs(a, df=12, degree=2, include_intercept=True)", {"a": a}, return_type='dataframe')
    spline_fit = sm.OLS(pred_i, spline_a).fit()
    spline_i = spline_fit.predict(spline_a)
    optimal_idx = int(np.argmax(pred_i))
    out = pd.DataFrame({
        '예산(억 원)': [budget_won/100_000_000],
        'A 비율': [f"{int(a[optimal_idx]*100)}%"],
        'B 비율': [f"{int(b[optimal_idx]*100)}%"],
        'Reach 1+(%)': [round(100.0 * pred_i[optimal_idx], 2)]
    })
    return a, pred_i, spline_i, out

# 최적화 UI
st.subheader("💰 예산 최적화")
tab1, tab2 = st.tabs(["예산 범위 최적화", "특정 예산 최적화"])

# 세션 상태 초기화
if "df_opt" not in st.session_state:
    st.session_state.df_opt = None
if "single_opt_result" not in st.session_state:
    st.session_state.single_opt_result = None
if "single_opt_curve" not in st.session_state:
    st.session_state.single_opt_curve = None

# 탭 1: 예산 범위 최적화
with tab1:
    st.session_state.max_units_val = st.slider("예산 범위(억 원)", 1, 30, 15, key="max_units_tab1_val")

    if st.button("예산 범위 최적화 실행", key="run_budget_range"):
        st.session_state.df_opt = optimize_mix_over_budget(
            cpm_a_input, cpm_b_input, max_budget_units=st.session_state.max_units_val
            )

    # 이전 실행 결과 유지
    if st.session_state.df_opt is not None:
        st.dataframe(st.session_state.df_opt)
        fig2, ax2 = plt.subplots()
        ax2.plot(st.session_state.df_opt['예산(억 원)'], st.session_state.df_opt['Reach 1+(%)'], marker='o')
        ax2.set_xlabel("Budget Range")
        ax2.set_ylabel("Reach 1+ (%)")
        ax2.grid(True, linestyle='--')
        st.pyplot(fig2, use_container_width=False)

# 탭 2: 특정 예산 최적화
with tab2:
    st.session_state.single_budget_val = st.number_input("특정 예산(억 원)", value=7.0, step=0.1, key="single_budget_tab2_val")
    
    if st.button("특정 예산 최적화 실행", key="run_single_budget"):
        a, pred_i, spline_i, out = optimize_single_budget(
            st.session_state.single_budget_val * 100_000_000, cpm_a_input, cpm_b_input
            )
        
        st.session_state.single_opt_result = out
        st.session_state.single_opt_curve = (a, pred_i, spline_i)

    # 이전 실행 결과 유지
    if st.session_state.single_opt_result is not None:
        a, pred_i, spline_i = st.session_state.single_opt_curve
        fig3, ax3 = plt.subplots()
        ax3.scatter(100*a, 100*pred_i, alpha=0.6, color='gold', s=30)
        ax3.plot(100*a, 100*spline_i, color='crimson')
        ax3.set_xlabel("Media A ratio(%)")
        ax3.set_ylabel("Reach 1+ (%)")
        ax3.grid(True, linestyle='--')
        st.pyplot(fig3, use_container_width=False)
        st.dataframe(st.session_state.single_opt_result)

