import streamlit as st
import time
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import statsmodels.api as sm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dropbox
from io import BytesIO

st.set_page_config(page_title="Reach 1/3+ Optimization")

APP_KEY        = st.secrets["dropbox"]["app_key"]
APP_SECRET     = st.secrets["dropbox"]["app_secret"]
REFRESH_TOKEN  = st.secrets["dropbox"]["refresh_token"]
DROPBOX_PATH   = st.secrets["dropbox"]["path"]
UNIVERSE_PATH  = "/Media Mix/universe.csv"

# ---------------------------
# Dropbox helpers
# ---------------------------
def dbx_client():
    return dropbox.Dropbox(
        oauth2_refresh_token=REFRESH_TOKEN,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
    )

def load_bytes_from_dropbox(path: str) -> BytesIO | None:
    try:
        dbx = dbx_client()
        _, res = dbx.files_download(path)
        return BytesIO(res.content)
    except Exception as e:
        st.error(f"⚠ Dropbox 파일 불러오기 실패 ({path}): {e}")
        return None

def load_image_from_dropbox(path: str) -> BytesIO | None:
    return load_bytes_from_dropbox(path)

def load_csv_from_dropbox(path: str, usecols=None, parse_dates=None) -> pd.DataFrame | None:
    bio = load_bytes_from_dropbox(path)
    if bio is None:
        return None
    # pyarrow → csv → utf-8 fallback
    try:
        return pd.read_csv(bio, engine="pyarrow", usecols=usecols, parse_dates=parse_dates)
    except Exception:
        bio.seek(0)
        try:
            return pd.read_csv(bio, usecols=usecols, low_memory=False, encoding="utf-8-sig", parse_dates=parse_dates)
        except Exception as e:
            st.error(f"⚠ CSV 불러오기 실패 ({path}): {e}")
            return None

# ---------------------------
# Title with logo
# ---------------------------
logo_bytes = load_image_from_dropbox("/Media Mix/logo.png")
col1, col2 = st.columns([1, 5])
with col1:
    if logo_bytes is not None:
        st.image(logo_bytes, use_container_width=True)
with col2:
    st.markdown("<h1> Reach 1/3+ 최적화</h1>", unsafe_allow_html=True)

# ---------------------------
# Load data
# ---------------------------
needed_cols = ['date', 'brand_id', 'target', 'media', 'impression', 'r1', 'r2', 'r3']
df_raw = load_csv_from_dropbox(DROPBOX_PATH, usecols=needed_cols, parse_dates=["date"])
if df_raw is None:
    st.stop()

msg_placeholder = st.empty()
msg_placeholder.success("✅ 데이터 불러오기 성공")
time.sleep(1)
msg_placeholder.empty()

df = df_raw[df_raw['r1'] != 0].copy()

metrics = ['impression','r1','r2','r3']
pivot = df.pivot_table(
    index=['date','brand_id','target'],
    columns='media',
    values=metrics,
    aggfunc='sum'
)
pivot.columns = [f"{a}_{b}" for a,b in pivot.columns.to_flat_index()]
pivot = pivot.reset_index()

rename_map = {
    'impression_total':'imps',
    'r1_total':'r1',
    'impression_tv':'imps_a',
    'r1_tv':'r1_a',
    'impression_digital':'imps_b',
    'r1_digital':'r1_b',
    'r2_total':'r2',
    'r2_tv':'r2_a',
    'r2_digital':'r2_b',
    'r3_total':'r3',
    'r3_tv':'r3_a',
    'r3_digital':'r3_b',
}
for src,dst in rename_map.items():
    if src in pivot.columns:
        pivot.rename(columns={src:dst}, inplace=True)

required_cols_total = ['imps','r1','imps_a','r1_a','imps_b','r1_b']
missing = [c for c in required_cols_total if c not in pivot.columns]
if missing:
    st.error(f"필수 데이터가 없습니다: {missing}")
    st.stop()

pivot_all    = pivot.copy()
pivot_strict = pivot.dropna(subset=required_cols_total).copy()
pivot_strict['r1_ab'] = pivot_strict['r1_a'] * pivot_strict['r1_b']

# Target select
target_list = sorted(pivot_strict['target'].unique())
if not target_list:
    st.error("⚠ 선택 가능한 타겟이 없습니다.")
    st.stop()
selected_target = st.selectbox("Target", target_list, index=0)

df_total = pivot_strict[pivot_strict['target'] == selected_target].reset_index(drop=True)
df_media = pivot_all[pivot_all['target'] == selected_target].reset_index(drop=True)
st.caption(f"✅ **{selected_target}** 데이터가 적용되었습니다.")

# ---------------------------
# Load universe for target
# ---------------------------
uni_df = load_csv_from_dropbox(UNIVERSE_PATH)
if uni_df is None or 'target' not in uni_df.columns or 'universe' not in uni_df.columns:
    st.error("⚠ universe.csv는 'target','universe' 컬럼을 포함해야 합니다.")
    st.stop()

row_uni = uni_df.loc[uni_df['target'] == selected_target, 'universe']
if row_uni.empty:
    st.warning("⚠ 선택한 타겟의 데이터를 찾지 못했습니다. 기본값을 사용합니다.")
    universe = float(uni_df['universe'].iloc[0])
else:
    universe = float(row_uni.iloc[0])

st.caption(f"👥 Universe: **{int(universe):,}**")

# ---------------------------
# Load CPRP default for target
# ---------------------------
CPRP_PATH = "/Media Mix/cprp.csv"

cprp_df = load_csv_from_dropbox(CPRP_PATH)
default_cprp_value = 1_000_000


if cprp_df is None or 'target' not in cprp_df.columns or 'cprp' not in cprp_df.columns:
    st.warning("⚠ 데이터를 찾지 못했습니다. 기본값을 사용합니다.")
    cprp_default_for_target = default_cprp_value
else:

    row_cprp = cprp_df.loc[cprp_df['target'] == selected_target, 'cprp']
    if row_cprp.empty:
        st.warning("⚠ 선택한 타겟의 CPRP 값을 찾지 못했습니다. 기본값을 사용합니다.")
        cprp_default_for_target = default_cprp_value
    else:
        try:
            cprp_default_for_target = float(row_cprp.iloc[0])
        except Exception:
            st.warning("⚠ 데이터를 찾지 못했습니다. 기본값을 사용합니다.")
            cprp_default_for_target = default_cprp_value

if 'last_target_for_cprp' not in st.session_state:
    st.session_state.last_target_for_cprp = None

if st.session_state.last_target_for_cprp != selected_target:

    st.session_state['cprp_input'] = f"{cprp_default_for_target:,.0f}"
    st.session_state.last_target_for_cprp = selected_target

# ---------------------------
# Prepare arrays
# ---------------------------
x_total = df_total['imps'].values
y_total = df_total['r1'].values
tv_mask_r1 = df_media[['imps_a','r1_a']].notna().all(axis=1)
dg_mask_r1 = df_media[['imps_b','r1_b']].notna().all(axis=1)
x_a  = df_media.loc[tv_mask_r1, 'imps_a'].values
y_a1 = df_media.loc[tv_mask_r1, 'r1_a'].values
x_b  = df_media.loc[dg_mask_r1, 'imps_b'].values
y_b1 = df_media.loc[dg_mask_r1, 'r1_b'].values

tv_mask_r2 = df_media[['imps_a','r2_a']].notna().all(axis=1)
dg_mask_r2 = df_media[['imps_b','r2_b']].notna().all(axis=1)
x_a2 = df_media.loc[tv_mask_r2, 'imps_a'].values
y_a2 = df_media.loc[tv_mask_r2, 'r2_a'].values
x_b2 = df_media.loc[dg_mask_r2, 'imps_b'].values
y_b2 = df_media.loc[dg_mask_r2, 'r2_b'].values

tv_mask_r3 = df_media[['imps_a','r3_a']].notna().all(axis=1)
dg_mask_r3 = df_media[['imps_b','r3_b']].notna().all(axis=1)
x_a3 = df_media.loc[tv_mask_r3, 'imps_a'].values
y_a3 = df_media.loc[tv_mask_r3, 'r3_a'].values
x_b3 = df_media.loc[dg_mask_r3, 'imps_b'].values
y_b3 = df_media.loc[dg_mask_r3, 'r3_b'].values

def hill(x, a, b, c):
    return c / (1.0 + (b / x)**a)

initial_params1 = [1.0, 25_000_000.0, 0.6]
initial_params2 = [1.0, 25_000_000.0, 0.3]
initial_params3 = [1.0, 25_000_000.0, 0.1]

bounds_a = ([0,0,0],[np.inf,np.inf,1.0])
bounds_b1 = ([0,0,0],[np.inf,np.inf,0.706])
bounds_b2 = ([0,0,0],[np.inf,np.inf,0.4])
bounds_b3 = ([0,0,0],[np.inf,np.inf,0.2])

popt_a1, _ = curve_fit(hill, x_a,  y_a1, p0=initial_params1, bounds=bounds_a, maxfev=20000)
popt_b1, _ = curve_fit(hill, x_b,  y_b1, p0=initial_params1, bounds=bounds_b1, maxfev=20000)
popt_a2, _ = curve_fit(hill, x_a2, y_a2, p0=initial_params1, bounds=bounds_a, maxfev=20000)
popt_b2, _ = curve_fit(hill, x_b2, y_b2, p0=initial_params2, bounds=bounds_b2, maxfev=20000)
popt_a3, _ = curve_fit(hill, x_a3, y_a3, p0=initial_params1, bounds=bounds_a, maxfev=20000)
popt_b3, _ = curve_fit(hill, x_b3, y_b3, p0=initial_params3, bounds=bounds_b3, maxfev=20000)

pred_a1_fit = hill(x_a, *popt_a1)
pred_b1_fit = hill(x_b, *popt_b1)

media_r1_result = pd.DataFrame({
    'Hill n (a)': [popt_a1[0], popt_b1[0]],
    'EC50 (b)':   [popt_a1[1], popt_b1[1]],
    'Max (c)':    [popt_a1[2], popt_b1[2]],
    'R-squared':  [r2_score(y_a1, pred_a1_fit), r2_score(y_b1, pred_b1_fit)],
    'MAE(%)':     [mean_absolute_error(y_a1, pred_a1_fit)*100, mean_absolute_error(y_b1, pred_b1_fit)*100]
}, index=['TV','Digital'])

X_train = pd.DataFrame({
    'r1_a': df_total['r1_a'].values,
    'r1_b': df_total['r1_b'].values,
    'r1_ab': df_total['r1_ab'].values
})
model_total = sm.OLS(y_total, X_train).fit()

B_A  = float(model_total.params['r1_a'])
B_B  = float(model_total.params['r1_b'])
B_AB = float(model_total.params['r1_ab'])

def predict_total_r1_np(r1_a, r1_b):

    return B_A * r1_a + B_B * r1_b + B_AB * (r1_a * r1_b)

# ---------------------------
# CPM/CPRP UI
# ---------------------------
def money_input(label, key, default=0.0, help=None, decimals=0, min_value=0.0):

    fmt = f"{{:,.{decimals}f}}"
    # 초기 값
    if key not in st.session_state:
        st.session_state[key] = fmt.format(default)

    # 입력창
    s = st.text_input(label, value=st.session_state[key], key=f"{key}_text", help=help)

    try:
        v = float(s.replace(",", ""))
        if v < min_value:
            raise ValueError
        st.session_state[key] = fmt.format(v)
    except ValueError:
        st.warning("숫자만 입력하세요. 예: 1,000,000")
        v = float(st.session_state[key].replace(",", ""))

    return v

col_cprp, col_cpm = st.columns(2)
with col_cprp:
    cprp_a_global = money_input(
        "TV CPRP(원)",
        key="cprp_input",
        default=cprp_default_for_target,
        help="천 단위 콤마(,)로 입력/표시됩니다.",
        decimals=0,
        min_value=0.0
    )
with col_cpm:
    cpm_b_global = money_input(
        "Digital CPM(원)",
        key="cpm_input",
        default=10_300.0,
        help="천 단위 콤마(,)로 입력/표시됩니다.",
        decimals=0,
        min_value=0.0
    )

# ---------------------------
# Cost to Impression (TV=CPRP, Digital=CPM)
# ---------------------------
def imps_from_tv_budget_by_cprp(budget_won, cprp_a, universe_val):

    budget = np.asarray(budget_won, dtype=float)
    cprp = float(cprp_a)
    uni  = float(universe_val)

    with np.errstate(divide='ignore', invalid='ignore'):
        imps = np.where((cprp > 0) & (uni > 0) & (budget > 0),
                        (budget / cprp) / 100 * uni,
                        0.0)

    if np.isscalar(budget_won):
        return float(imps)
    return imps

def imps_from_digital_budget_by_cpm(budget_won, cpm_b):

    budget = np.asarray(budget_won, dtype=float)
    cpm = float(cpm_b)

    with np.errstate(divide='ignore', invalid='ignore'):
        imps = np.where((cpm > 0) & (budget > 0),
                        budget / (cpm / 1000.0),
                        0.0)

    if np.isscalar(budget_won):
        return float(imps)
    return imps

# ---------------------------
# Functions
# ---------------------------
UNIT = 100_000_000

def plateau_after_exceed(arr, threshold=1.0):
    a = np.asarray(arr, dtype=float).copy()
    over = a > threshold
    if np.any(over):
        i = int(np.argmax(over))
        if i > 0:
            a[i:] = a[i-1]
        else:
            a[:] = threshold
    return a

def analyze_custom_budget1(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    a_won = a_eok * unit
    b_won = b_eok * unit

    a_imps = imps_from_tv_budget_by_cprp(a_won, cprp_a, universe_val)
    b_imps = imps_from_digital_budget_by_cpm(b_won, cpm_b)

    a_r1 = hill(np.array([a_imps]), *popt_a1) if a_imps > 0 else np.array([0.0])
    b_r1 = hill(np.array([b_imps]), *popt_b1) if b_imps > 0 else np.array([0.0])

    if a_won > 0 and b_won == 0:
        total_r1 = a_r1.copy()
    elif b_won > 0 and a_won == 0:
        total_r1 = b_r1.copy()
    else:
        total_r1 = predict_total_r1_np(a_r1, b_r1)
        if np.isscalar(total_r1):
            total_r1 = np.array([total_r1], dtype=float)

    df_out = pd.DataFrame({
        '항목': ['TV(억 원)','Digital(억 원)','총(억 원)','TV Reach 1+(%)','Digital Reach 1+(%)','Total Reach 1+(%)'],
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

def analyze_custom_budget3(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    a_won = a_eok * unit
    b_won = b_eok * unit

    a_imps = imps_from_tv_budget_by_cprp(a_won, cprp_a, universe_val)
    b_imps = imps_from_digital_budget_by_cpm(b_won, cpm_b)

    a_r1 = hill(np.array([a_imps]), *popt_a1) if a_imps > 0 else np.array([0.0])
    a_r2 = hill(np.array([a_imps]), *popt_a2) if a_imps > 0 else np.array([0.0])
    a_r3 = hill(np.array([a_imps]), *popt_a3) if a_imps > 0 else np.array([0.0])

    b_r1 = hill(np.array([b_imps]), *popt_b1) if b_imps > 0 else np.array([0.0])
    b_r2 = hill(np.array([b_imps]), *popt_b2) if b_imps > 0 else np.array([0.0])
    b_r3 = hill(np.array([b_imps]), *popt_b3) if b_imps > 0 else np.array([0.0])

    if a_won > 0 and b_won == 0:
        total_r1 = a_r1.copy()
    elif b_won > 0 and a_won == 0:
        total_r1 = b_r1.copy()
    else:
        total_r1 = predict_total_r1_np(a_r1, b_r1)
        if np.isscalar(total_r1):
            total_r1 = np.array([total_r1], dtype=float)

    a_r0_ = 1.0 - a_r1; a_r1_ = a_r1 - a_r2; a_r2_ = a_r2 - a_r3
    b_r0_ = 1.0 - b_r1; b_r1_ = b_r1 - b_r2; b_r2_ = b_r2 - b_r3

    total_r2 = total_r1 - (a_r1_ * b_r0_ + b_r1_ * a_r0_)
    total_r3 = total_r2 - (a_r2_ * b_r0_ + b_r2_ * a_r0_ + a_r1_ * b_r1_)

    if a_won > 0 and b_won == 0:
        total_r3 = a_r3.copy()
    elif b_won > 0 and a_won == 0:
        total_r3 = b_r3.copy()

    df_out = pd.DataFrame({
        '항목': ['TV(억 원)','Digital(억 원)','총(억 원)','TV Reach 3+(%)','Digital Reach 3+(%)','Total Reach 3+(%)'],
        '값': [
            np.round(a_won/unit, 2), np.round(b_won/unit, 2), np.round((a_won+b_won)/unit, 2),
            np.round(100*a_r3[0], 2), np.round(100*b_r3[0], 2), np.round(100*total_r3[0], 2)
        ]
    })
    parts = {'a_r3': a_r3, 'b_r3': b_r3, 'total_r3': total_r3}
    return df_out, parts

def optimize_total_budget1(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    total_won = (a_eok + b_eok) * unit
    a_share = np.arange(0, 101, dtype=np.float64) / 100.0
    b_share = 1.0 - a_share

    a_imps = imps_from_tv_budget_by_cprp(a_share * total_won, cprp_a, universe_val)
    b_imps = imps_from_digital_budget_by_cpm(b_share * total_won, cpm_b)

    a_r1_curve = hill(a_imps, *popt_a1)
    b_r1_curve = hill(b_imps, *popt_b1)

    total_r1_curve_raw = predict_total_r1_np(a_r1_curve, b_r1_curve)
    total_r1_curve = plateau_after_exceed(total_r1_curve_raw, threshold=1.0)

    idx1 = int(np.argmax(total_r1_curve))
    total_r1_value = (a_r1_curve[idx1] if a_share[idx1] >= 0.99
                      else b_r1_curve[idx1] if b_share[idx1] >= 0.99
                      else total_r1_curve[idx1])

    return {
        'a_share': float(a_share[idx1]), 'b_share': float(b_share[idx1]),
        'a_r1': float(a_r1_curve[idx1]), 'b_r1': float(b_r1_curve[idx1]),
        'total_r1': float(total_r1_value),
    }

def optimize_total_budget3(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    total_won = (a_eok + b_eok) * unit
    a3_share = np.arange(0, 101, dtype=np.float64) / 100.0
    b3_share = 1.0 - a3_share

    a_imps = imps_from_tv_budget_by_cprp(a3_share * total_won, cprp_a, universe_val)
    b_imps = imps_from_digital_budget_by_cpm(b3_share * total_won, cpm_b)

    a_r1_curve = hill(a_imps, *popt_a1); a_r2_curve = hill(a_imps, *popt_a2); a_r3_curve = hill(a_imps, *popt_a3)
    b_r1_curve = hill(b_imps, *popt_b1); b_r2_curve = hill(b_imps, *popt_b2); b_r3_curve = hill(b_imps, *popt_b3)

    total_r1_curve_raw = predict_total_r1_np(a_r1_curve, b_r1_curve)
    total_r1_curve = plateau_after_exceed(total_r1_curve_raw, threshold=1.0)

    a_r0_ = 1 - a_r1_curve; a_r1_ = a_r1_curve - a_r2_curve; a_r2_ = a_r2_curve - a_r3_curve
    b_r0_ = 1 - b_r1_curve; b_r1_ = b_r1_curve - b_r2_curve; b_r2_ = b_r2_curve - b_r3_curve
    total_r2_curve = total_r1_curve - (a_r1_ * b_r0_ + b_r1_ * a_r0_)
    total_r3_curve = total_r2_curve - (a_r2_ * b_r0_ + b_r2_ * a_r0_ + a_r1_ * b_r1_)

    idx3 = int(np.argmax(total_r3_curve))
    total_r3_value = (a_r3_curve[idx3] if a3_share[idx3] >= 0.99
                      else b_r3_curve[idx3] if b3_share[idx3] >= 0.99
                      else total_r3_curve[idx3])

    return {
        'a3_share': float(a3_share[idx3]), 'b3_share': float(b3_share[idx3]),
        'a_r3': float(a_r3_curve[idx3]), 'b_r3': float(b_r3_curve[idx3]),
        'total_r3': float(total_r3_value),
    }

def compare_user_vs_opt1(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    user_df, user_parts = analyze_custom_budget1(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit)
    user_a_r1 = float(user_parts['a_r1'][0])
    user_b_r1 = float(user_parts['b_r1'][0])
    user_total_r1 = float(user_parts['total_r1'][0])

    opt = optimize_total_budget1(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit)
    total_eok = a_eok + b_eok
    a_eok_opt = round(total_eok * opt['a_share'], 2)
    b_eok_opt = round(total_eok * opt['b_share'], 2)

    summary1 = pd.DataFrame([
        {
            '구분': '사용자안',
            'TV 예산(억 원)': round(a_eok, 2),
            'Digital 예산(억 원)': round(b_eok, 2),
            'TV 비중': f"{int(round(100 * (a_eok / total_eok))) if total_eok>0 else 0}%",
            'Digital 비중': f"{int(round(100 * (b_eok / total_eok))) if total_eok>0 else 0}%",
            'TV Reach 1+(%)': round(100 * user_a_r1, 2),
            'Digital Reach 1+(%)': round(100 * user_b_r1, 2),
            'Total Reach 1+(%)': round(100 * user_total_r1, 2),
        },
        {
            '구분': '최적화안',
            'TV 예산(억 원)': a_eok_opt,
            'Digital 예산(억 원)': b_eok_opt,
            'TV 비중': f"{int(round(100 * opt['a_share']))}%",
            'Digital 비중': f"{int(round(100 * opt['b_share']))}%",
            'TV Reach 1+(%)': round(100 * opt['a_r1'], 2),
            'Digital Reach 1+(%)': round(100 * opt['b_r1'], 2),
            'Total Reach 1+(%)': round(100 * opt['total_r1'], 2),
        }
    ])
    return summary1

def compare_user_vs_opt3(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit=UNIT):
    user_df, user_parts = analyze_custom_budget3(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit)
    user_a_r3 = float(user_parts['a_r3'][0])
    user_b_r3 = float(user_parts['b_r3'][0])
    user_total_r3 = float(user_parts['total_r3'][0])

    opt = optimize_total_budget3(a_eok, b_eok, cprp_a, cpm_b, universe_val, unit)
    total_eok = a_eok + b_eok
    a_eok_opt = round(total_eok * opt['a3_share'], 2)
    b_eok_opt = round(total_eok * opt['b3_share'], 2)

    summary3 = pd.DataFrame([
        {
            '구분': '사용자안',
            'TV 예산(억 원)': round(a_eok, 2),
            'Digital 예산(억 원)': round(b_eok, 2),
            'TV 비중': f"{int(round(100 * (a_eok / total_eok))) if total_eok>0 else 0}%",
            'Digital 비중': f"{int(round(100 * (b_eok / total_eok))) if total_eok>0 else 0}%",
            'TV Reach 3+(%)': round(100 * user_a_r3, 2),
            'Digital Reach 3+(%)': round(100 * user_b_r3, 2),
            'Total Reach 3+(%)': round(100 * user_total_r3, 2),
        },
        {
            '구분': '최적화안',
            'TV 예산(억 원)': a_eok_opt,
            'Digital 예산(억 원)': b_eok_opt,
            'TV 비중': f"{int(round(100 * opt['a3_share']))}%",
            'Digital 비중': f"{int(round(100 * opt['b3_share']))}%",
            'TV Reach 3+(%)': round(100 * opt['a_r3'], 2),
            'Digital Reach 3+(%)': round(100 * opt['b_r3'], 2),
            'Total Reach 3+(%)': round(100 * opt['total_r3'], 2),
        }
    ])
    return summary3

def optimize_mix_over_budget1(cprp_a, cpm_b, universe_val, max_budget_units=20, unit=UNIT):
    a_share = np.arange(0,101,dtype=np.float64)/100.0
    b_share = 1.0 - a_share

    budget_eok = np.arange(0, max_budget_units+1)
    budget_won = budget_eok * unit

    a_imps_only = imps_from_tv_budget_by_cprp(budget_won, cprp_a, universe_val)
    b_imps_only = imps_from_digital_budget_by_cpm(budget_won, cpm_b)
    only_a1 = hill(a_imps_only, *popt_a1)
    only_b1 = hill(b_imps_only, *popt_b1)
    df_only1_full = pd.DataFrame({'예산(억 원)': budget_eok,
                                  'Only TV': np.round(100*only_a1, 2),
                                  'Only Digital': np.round(100*only_b1, 2)})

    results1, total_r1_raw = [], []
    for won, eok in zip(budget_won, budget_eok):
        a_budget = a_share * won
        b_budget = b_share * won

        a_imps = imps_from_tv_budget_by_cprp(a_budget, cprp_a, universe_val)
        b_imps = imps_from_digital_budget_by_cpm(b_budget, cpm_b)

        a_r1 = hill(a_imps, *popt_a1)
        b_r1 = hill(b_imps, *popt_b1)

        total_r1_curve_raw = predict_total_r1_np(a_r1, b_r1)
        total_r1_curve = plateau_after_exceed(total_r1_curve_raw, threshold=1.0)

        idx1 = int(np.argmax(total_r1_curve))
        best_total_r1 = (a_r1[idx1] if a_share[idx1] >= 0.99
                         else b_r1[idx1] if b_share[idx1] >= 0.99
                         else total_r1_curve[idx1])

        total_r1_raw.append(best_total_r1)
        results1.append({'예산(억 원)': eok,
                         'TV 비중': f"{int(a_share[idx1]*100)}%",
                         'Digital 비중': f"{int(b_share[idx1]*100)}%"})

    total_r1 = np.round(100.0 * np.clip(np.array(total_r1_raw), 0.0, 1.0), 2)

    df_opt1_full = pd.DataFrame(results1)
    df_opt1_full['Total Reach 1+(%)'] = total_r1
    df_only1 = df_only1_full[df_only1_full['예산(억 원)']>0].reset_index(drop=True)
    df_opt1  = df_opt1_full[df_opt1_full['예산(억 원)']>0].reset_index(drop=True)
    return df_opt1_full, df_only1_full, df_opt1, df_only1

def optimize_mix_over_budget3(cprp_a, cpm_b, universe_val, max_budget_units=20, unit=UNIT):
    a3_share = np.arange(0,101,dtype=np.float64)/100.0
    b3_share = 1.0 - a3_share

    budget_eok = np.arange(0, max_budget_units+1)
    budget_won = budget_eok * unit

    a_imps_only = imps_from_tv_budget_by_cprp(budget_won, cprp_a, universe_val)
    b_imps_only = imps_from_digital_budget_by_cpm(budget_won, cpm_b)
    only_a3 = hill(a_imps_only, *popt_a3)
    only_b3 = hill(b_imps_only, *popt_b3)
    df_only3_full = pd.DataFrame({'예산(억 원)': budget_eok,
                                  'Only TV': np.round(100*only_a3, 2),
                                  'Only Digital': np.round(100*only_b3, 2)})

    results3, total_r3_raw = [], []
    for won, eok in zip(budget_won, budget_eok):
        a_budget = a3_share * won
        b_budget = b3_share * won

        a_imps = imps_from_tv_budget_by_cprp(a_budget, cprp_a, universe_val)
        b_imps = imps_from_digital_budget_by_cpm(b_budget, cpm_b)

        a_r1 = hill(a_imps, *popt_a1); a_r2 = hill(a_imps, *popt_a2); a_r3 = hill(a_imps, *popt_a3)
        b_r1 = hill(b_imps, *popt_b1); b_r2 = hill(b_imps, *popt_b2); b_r3 = hill(b_imps, *popt_b3)

        total_r1_curve_raw = predict_total_r1_np(a_r1, b_r1)
        total_r1_curve = plateau_after_exceed(total_r1_curve_raw, threshold=1.0)

        a_r0_ = 1 - a_r1; a_r1_ = a_r1 - a_r2; a_r2_ = a_r2 - a_r3
        b_r0_ = 1 - b_r1; b_r1_ = b_r1 - b_r2; b_r2_ = b_r2 - b_r3
        total_r2_curve = total_r1_curve - (a_r1_ * b_r0_ + b_r1_ * a_r0_)
        total_r3_curve = total_r2_curve - (a_r2_ * b_r0_ + b_r2_ * a_r0_ + a_r1_ * b_r1_)

        idx3 = int(np.argmax(total_r3_curve))
        best_total_r3 = (a_r3[idx3] if a3_share[idx3] >= 0.99
                         else b_r3[idx3] if b3_share[idx3] >= 0.99
                         else total_r3_curve[idx3])

        total_r3_raw.append(best_total_r3)
        results3.append({'예산(억 원)': eok,
                         'TV 비중': f"{int(a3_share[idx3]*100)}%",
                         'Digital 비중': f"{int(b3_share[idx3]*100)}%"})

    total_r3 = np.round(100.0 * np.clip(np.array(total_r3_raw), 0.0, 1.0), 2)

    df_opt3_full = pd.DataFrame(results3)
    df_opt3_full['Total Reach 3+(%)'] = total_r3
    df_only3 = df_only3_full[df_only3_full['예산(억 원)']>0].reset_index(drop=True)
    df_opt3  = df_opt3_full[df_opt3_full['예산(억 원)']>0].reset_index(drop=True)
    return df_opt3_full, df_only3_full, df_opt3, df_only3

# ---------------------------
# UI: Pages (Reach1 / Reach3)
# ---------------------------
page1, page3 = st.tabs(["Reach 1+ 최적화", "Reach 3+ 최적화"])

# 공통: 세션 키 초기화
for key in [
    "r1_compare_result", "r1_single_curve", "r1_single_out",
    "r1_sweep_opt_full", "r1_sweep_only_full", "r1_sweep_opt", "r1_sweep_only",
    "r3_compare_result", "r3_single_curve", "r3_single_out",
    "r3_sweep_opt_full", "r3_sweep_only_full", "r3_sweep_opt", "r3_sweep_only"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ===========================
# Reach 1+ Page
# ===========================
with page1:
    tab1_1, tab1_2, tab1_3 = st.tabs(["개별 예산 최적화", "총 예산 최적화", "예산 범위 최적화"])

    # --- 개별 예산 최적화 (사용자안 vs 최적화안) ---
    with tab1_1:
        c_a, c_b = st.columns([1, 1])
        with c_a:
            a_eok_input = st.number_input("TV 예산(억 원)", value=3.5, step=0.1, min_value=0.0)
        with c_b:
            b_eok_input = st.number_input("Digital 예산(억 원)", value=3.5, step=0.1, min_value=0.0)

        if st.button("실행", type="primary", key="r1_compare_run"):
            summary_df = compare_user_vs_opt1(a_eok_input, b_eok_input, cprp_a_global, cpm_b_global, universe)
            st.session_state.r1_compare_result = summary_df

        if st.session_state.r1_compare_result is not None:
            summary_df = st.session_state.r1_compare_result
            summary_wide = summary_df.set_index('구분').T.rename_axis('구분')
            summary_wide = summary_wide[['사용자안', '최적화안']]

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

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=labels, y=user_vals, name='User', marker_color='gold',
                                  text=[f"<b>{v:.2f}%<b>" for v in user_vals],
                                  textposition='outside', textfont_size=12, textfont_color="black", hoverinfo='skip'))
            fig1.add_trace(go.Bar(x=labels, y=opt_vals, name='Opt', marker_color='#003594',
                                  text=[f"<b>{v:.2f}%<b>" for v in opt_vals],
                                  textposition='outside', textfont_size=12, textfont_color="black", hoverinfo='skip'))
            fig1.update_layout(
                barmode='group',
                yaxis=dict(range=[0, 100], title="Reach 1+(%)"),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                dragmode=False,
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                height=400,
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.dataframe(summary_wide, use_container_width=True)

    # --- 총 예산 최적화(비중 곡선) ---
    with tab1_2:
        total_eok_input = st.number_input("총 예산(억 원)", value=7.0, step=0.1, min_value=0.0)
        if st.button("실행", type="primary", key="r1_single_run"):
            a = np.arange(0, 101, dtype=np.float64) / 100.0
            b = 1.0 - a
            won = total_eok_input * UNIT

            a_budget = a * won
            b_budget = b * won
            a_imps = imps_from_tv_budget_by_cprp(a_budget, cprp_a_global, universe)
            b_imps = imps_from_digital_budget_by_cpm(b_budget, cpm_b_global)

            a_r1 = hill(a_imps, *popt_a1)
            b_r1 = hill(b_imps, *popt_b1)

            pred_raw = predict_total_r1_np(a_r1, b_r1)
            pred = plateau_after_exceed(pred_raw, threshold=1.0)

            best_idx = int(np.argmax(pred))
            
            best_total_r1 = (
                a_r1[best_idx] if a[best_idx] >= 0.99
                else b_r1[best_idx] if (1.0 - a[best_idx]) >= 0.99
                else pred[best_idx]
            )

            st.session_state.r1_single_curve = (a, pred)
            out = pd.DataFrame({
                'TV 비중': [f"{int(a[best_idx]*100)}%"],
                'Digital 비중': [f"{int((1.0-a[best_idx])*100)}%"],
                'Total Reach 1+(%)': [round(100.0 * float(best_total_r1), 2)]
            })
            st.session_state.r1_single_out = out

        if st.session_state.r1_single_curve is not None:
            a, pred = st.session_state.r1_single_curve
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=100*a, y=out['Total Reach 1+(%)'], mode='lines+markers',
                                      name='Predicted', marker=dict(size=4, color='#003594')))
            fig2.update_layout(
                xaxis=dict(title='TV ratio (%)', range=[0, 100]),
                yaxis=dict(title='Reach 1+(%)'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x', template="plotly_white",
                width=700, height=400, dragmode=False,
                legend=dict(yanchor="top", y=1, xanchor="left", x=0),
            )
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
            st.plotly_chart(fig2, use_container_width=True)

        if st.session_state.r1_single_out is not None:
            st.dataframe(st.session_state.r1_single_out, use_container_width=True)

    # --- 예산 범위 최적화 ---
    with tab1_3:
        max_units = st.slider("예산 범위(억 원)", min_value=1, max_value=20, value=10, key="r1_max_units")
        if st.button("실행", type="primary", key="r1_sweep_run"):
            df_opt1_full, df_only1_full, df_opt1, df_only1 = optimize_mix_over_budget1(cprp_a_global, cpm_b_global, universe, max_budget_units=max_units)
            st.session_state.r1_sweep_opt_full = df_opt1_full
            st.session_state.r1_sweep_only_full = df_only1_full
            st.session_state.r1_sweep_opt = df_opt1
            st.session_state.r1_sweep_only = df_only1

        if (st.session_state.r1_sweep_opt_full is not None) and (st.session_state.r1_sweep_only_full is not None):
            df_opt_full = st.session_state.r1_sweep_opt_full
            df_only_full = st.session_state.r1_sweep_only_full
            df_opt  = st.session_state.r1_sweep_opt

            fig3 = go.Figure(layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
            tv=df_opt_full['TV 비중'].astype(str)
            digital=df_opt_full['Digital 비중'].astype(str)
            customdata = np.column_stack([tv.values, digital.values])

            fig3.add_trace(go.Scatter(
                x=df_opt_full['예산(억 원)'], y=df_opt_full['Total Reach 1+(%)'],
                mode='lines+markers', name='Opt Mix',
                customdata=customdata, marker=dict(color='#003594'),
                hovertemplate='TV: %{customdata[0]}<br>Digital: %{customdata[1]}<br>Reach 1+: %{y:.2f}%'
            ))
            fig3.add_trace(go.Scatter(
                x=df_only_full['예산(억 원)'], y=df_only_full['Only TV'],
                mode='lines+markers', name='Only TV', marker=dict(color='#ff7473'),
                hovertemplate='Reach 1+: %{y:.2f}%'
            ))
            fig3.add_trace(go.Scatter(
                x=df_only_full['예산(억 원)'], y=df_only_full['Only Digital'],
                mode='lines+markers', name='Only Digital', marker=dict(color='gold'),
                hovertemplate='Reach 1+: %{y:.2f}%'
            ))
            fig3.update_layout(
                xaxis_title="예산(억 원)", yaxis_title="Reach 1+(%)",
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'),
                hovermode='x', template='plotly_white',
                width=700, height=500, dragmode=False,
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(df_opt, use_container_width=True)

# ===========================
# Reach 3+ Page
# ===========================
with page3:
    tab3_1, tab3_2, tab3_3 = st.tabs(["개별 예산 최적화", "총 예산 최적화", "예산 범위 최적화"])

    # --- 개별 예산 최적화 (사용자안 vs 최적화안) ---
    with tab3_1:
        c_a, c_b = st.columns([1, 1])
        with c_a:
            a_eok_input3 = st.number_input("TV 예산(억 원)", value=3.5, step=0.1, min_value=0.0, key="r3_tv_eok")
        with c_b:
            b_eok_input3 = st.number_input("Digital 예산(억 원)", value=3.5, step=0.1, min_value=0.0, key="r3_dg_eok")

        if st.button("실행", type="primary", key="r3_compare_run"):
            summary_df3 = compare_user_vs_opt3(a_eok_input3, b_eok_input3, cprp_a_global, cpm_b_global, universe)
            st.session_state.r3_compare_result = summary_df3

        if st.session_state.r3_compare_result is not None:
            summary_df3 = st.session_state.r3_compare_result
            summary_wide3 = summary_df3.set_index('구분').T.rename_axis('구분')
            summary_wide3 = summary_wide3[['사용자안', '최적화안']]

            labels = ['TV', 'Digital', 'Total']
            user_vals = [
                summary_wide3.loc['TV Reach 3+(%)', '사용자안'],
                summary_wide3.loc['Digital Reach 3+(%)', '사용자안'],
                summary_wide3.loc['Total Reach 3+(%)', '사용자안'],
            ]
            opt_vals = [
                summary_wide3.loc['TV Reach 3+(%)', '최적화안'],
                summary_wide3.loc['Digital Reach 3+(%)', '최적화안'],
                summary_wide3.loc['Total Reach 3+(%)', '최적화안'],
            ]

            fig31 = go.Figure()
            fig31.add_trace(go.Bar(x=labels, y=user_vals, name='User', marker_color='gold',
                                  text=[f"<b>{v:.2f}%<b>" for v in user_vals],
                                  textposition='outside', textfont_size=12, textfont_color="black", hoverinfo='skip'))
            fig31.add_trace(go.Bar(x=labels, y=opt_vals, name='Opt', marker_color='#003594',
                                  text=[f"<b>{v:.2f}%<b>" for v in opt_vals],
                                  textposition='outside', textfont_size=12, textfont_color="black", hoverinfo='skip'))
            fig31.update_layout(
                barmode='group',
                yaxis=dict(range=[0, 100], title="Reach 3+(%)"),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                dragmode=False,
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                height=400,
            )
            st.plotly_chart(fig31, use_container_width=True)
            st.dataframe(summary_wide3, use_container_width=True)

    with tab3_2:
        total_eok_input = st.number_input("총 예산(억 원)", value=7.0, step=0.1, min_value=0.0, key="r3_total_eok")
        if st.button("실행", type="primary", key="r3_single_run"):
            a = np.arange(0, 101, dtype=np.float64) / 100.0
            b = 1.0 - a
            won = total_eok_input * UNIT

            a_budget = a * won
            b_budget = b * won
            a_imps = imps_from_tv_budget_by_cprp(a_budget, cprp_a_global, universe)
            b_imps = imps_from_digital_budget_by_cpm(b_budget, cpm_b_global)

            a_r1 = hill(a_imps, *popt_a1)
            b_r1 = hill(b_imps, *popt_b1)
            a_r2 = hill(a_imps, *popt_a2)
            b_r2 = hill(b_imps, *popt_b2)
            a_r3 = hill(a_imps, *popt_a3)
            b_r3 = hill(b_imps, *popt_b3)
            a_r0 = 1 - a_r1
            b_r0 = 1 - b_r1
            a_r1_ = a_r1 - a_r2
            b_r1_ = b_r1 - b_r2
            a_r2_ = a_r2 - a_r3
            b_r2_ = b_r2 - b_r3

            pred_r1_raw = predict_total_r1_np(a_r1, b_r1)
            pred_r1 = plateau_after_exceed(pred_r1_raw, threshold=1.0)
            pred_r2 = pred_r1 - (a_r1_ * b_r0 + b_r1_ * a_r0)
            pred_r3 = pred_r2 - (a_r2_ * b_r0 + b_r2_ * a_r0 + a_r1_ * b_r1_)

            best_idx = int(np.argmax(pred_r3))
            
            best_total_r3 = (
                a_r3[best_idx] if a[best_idx] >= 0.99
                else b_r3[best_idx] if (1.0 - a[best_idx]) >= 0.99
                else pred_r3[best_idx]
            )

            st.session_state.r3_single_curve = (a, pred_r3)
            out = pd.DataFrame({
                'TV 비중': [f"{int(a[best_idx]*100)}%"],
                'Digital 비중': [f"{int((1.0-a[best_idx])*100)}%"],
                'Total Reach 3+(%)': [round(100.0 * float(best_total_r3), 2)]
            })
            st.session_state.r3_single_out = out

        if st.session_state.r3_single_curve is not None:
            a, pred_r3 = st.session_state.r3_single_curve
            fig32 = go.Figure()
            fig32.add_trace(go.Scatter(x=100*a, y=out['Total Reach 3+(%)'], mode='lines+markers',
                                      name='Predicted', marker=dict(size=4, color='#003594')))
            fig32.update_layout(
                xaxis=dict(title='TV ratio (%)', range=[0, 100]),
                yaxis=dict(title='Reach 3+(%)'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x', template="plotly_white",
                width=700, height=400, dragmode=False,
                legend=dict(yanchor="top", y=1, xanchor="left", x=0),
            )
            fig32.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
            st.plotly_chart(fig32, use_container_width=True)

        if st.session_state.r3_single_out is not None:
            st.dataframe(st.session_state.r3_single_out, use_container_width=True)

    # --- 예산 범위 최적화(스윕) ---
    with tab3_3:
        max_units3 = st.slider("예산 범위(억 원)", min_value=1, max_value=20, value=10, key="r3_max_units")
        if st.button("실행", type="primary", key="r3_sweep_run"):
            df_opt_full3, df_only_full3, df_opt3, df_only3 = optimize_mix_over_budget3(cprp_a_global, cpm_b_global, universe, max_budget_units=max_units3)
            st.session_state.r3_sweep_opt_full = df_opt_full3
            st.session_state.r3_sweep_only_full = df_only_full3
            st.session_state.r3_sweep_opt = df_opt3
            st.session_state.r3_sweep_only = df_only3

        if (st.session_state.r3_sweep_opt_full is not None) and (st.session_state.r3_sweep_only_full is not None):
            df_opt_full3 = st.session_state.r3_sweep_opt_full
            df_only_full3 = st.session_state.r3_sweep_only_full
            df_opt3  = st.session_state.r3_sweep_opt

            fig33 = go.Figure(layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
            tv=df_opt_full3['TV 비중'].astype(str)
            digital=df_opt_full3['Digital 비중'].astype(str)
            customdata = np.column_stack([tv.values, digital.values])

            fig33.add_trace(go.Scatter(
                x=df_opt_full3['예산(억 원)'], y=df_opt_full3['Total Reach 3+(%)'],
                mode='lines+markers', name='Opt Mix',
                customdata=customdata, marker=dict(color='#003594'),
                hovertemplate='TV: %{customdata[0]}<br>Digital: %{customdata[1]}<br>Reach 3+: %{y:.2f}%'
            ))
            fig33.add_trace(go.Scatter(
                x=df_only_full3['예산(억 원)'], y=df_only_full3['Only TV'],
                mode='lines+markers', name='Only TV', marker=dict(color='#ff7473'),
                hovertemplate='Reach 3+: %{y:.2f}%'
            ))
            fig33.add_trace(go.Scatter(
                x=df_only_full3['예산(억 원)'], y=df_only_full3['Only Digital'],
                mode='lines+markers', name='Only Digital', marker=dict(color='gold'),
                hovertemplate='Reach 3+: %{y:.2f}%'
            ))
            fig33.update_layout(
                xaxis_title="예산(억 원)", yaxis_title="Reach 3+(%)",
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'),
                hovermode='x', template='plotly_white',
                width=700, height=500, dragmode=False,
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig33, use_container_width=True)
            st.dataframe(df_opt3, use_container_width=True)

