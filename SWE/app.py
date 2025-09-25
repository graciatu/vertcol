import os
import uuid
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, request, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # 서버에서 GUI 없이 렌더
import matplotlib.pyplot as plt

# -----------------------------
# Flask & data
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ensure plots dir
PLOTS_DIR = os.path.join(app.static_folder, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
verts = pd.read_excel("verts.xlsx")
verts.columns = [c.strip() for c in verts.columns]

VERTE_NAMES = ["C2","C3","C4","C5","C6","C7",
               "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
               "L1","L2","L3","L4","L5"]

# Sum_Verts 보장
if "Sum_Verts" not in verts.columns:
    missing = [v for v in VERTE_NAMES if v not in verts.columns]
    if missing:
        raise ValueError(f"Missing columns in verts.xlsx: {missing}")
    verts["Sum_Verts"] = verts[VERTE_NAMES].sum(axis=1)

# Sex 보장
if "Sex" not in verts.columns:
    verts["Sex"] = "UD"

# -----------------------------
# Helpers
# -----------------------------
def sex_key_from_form(form_value: str) -> str:
    s = (form_value or "").strip().lower()
    if s.startswith("male"):
        return "M"
    if s.startswith("female"):
        return "F"
    return "UD"

def get_sex_filtered_df(sex_key: str) -> pd.DataFrame:
    if sex_key in ("M","F","UD"):
        df = verts[verts["Sex"] == sex_key]
        if len(df) == 0 and sex_key == "UD":
            df = verts.copy()
        return df.copy()
    return verts.copy()

def build_training_df(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    cols = ["Sum_Verts"] + predictors
    return df.loc[:, cols].dropna()

def safe_k(n: int, k_default: int = 5) -> int:
    if n < 4:
        return 0
    return min(k_default, max(2, min(n, n // 2)))

def cv_run(train_df: pd.DataFrame, predictors: list, method: str = "ols", k: int = 5, seed: int = 42):
    n = len(train_df)
    k_eff = safe_k(n, k_default=k)
    if k_eff < 2:
        return {"summary": {"k": None, "R2_mean": np.nan, "RMSE_mean": np.nan},
                "details": []}

    kf = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
    r2_list, rmse_list = [], []

    X_all = train_df[predictors]
    y_all = train_df["Sum_Verts"].values

    for tr_idx, te_idx in kf.split(X_all):
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        if method == "ols":
            X_tr_ = sm.add_constant(X_tr, has_constant="add")
            X_te_ = sm.add_constant(X_te, has_constant="add")
            fit = sm.OLS(y_tr, X_tr_).fit()
            y_hat = fit.predict(X_te_)
        else:
            rf = RandomForestRegressor(n_estimators=500, random_state=seed)
            rf.fit(X_tr, y_tr)
            y_hat = rf.predict(X_te)

        r2 = r2_score(y_te, y_hat)
        rmse = mean_squared_error(y_te, y_hat, squared=False)
        r2_list.append(r2)
        rmse_list.append(rmse)

    return {
        "summary": {
            "k": k_eff,
            "R2_mean": float(np.nanmean(r2_list)),
            "R2_sd": float(np.nanstd(r2_list, ddof=1)) if len(r2_list) > 1 else np.nan,
            "RMSE_mean": float(np.nanmean(rmse_list)),
            "RMSE_sd": float(np.nanstd(rmse_list, ddof=1)) if len(rmse_list) > 1 else np.nan
        },
        "details": [{"fold": i+1, "R2": r2_list[i], "RMSE": rmse_list[i]} for i in range(len(r2_list))]
    }

def fit_and_predict(user_values: dict, sex_key: str, method: str = "ols", k: int = 5, seed: int = 42):
    """
    user_values: {"C2": 34.2, "T1": 28.4, ...}  (0/빈값 제외한 실제 입력만 포함)
    """
    use_vars = [v for v in user_values.keys() if v in VERTE_NAMES and pd.notnull(user_values[v]) and float(user_values[v]) != 0]
    if len(use_vars) == 0:
        raise ValueError("No valid vertebrae provided by user.")

    df_sex = get_sex_filtered_df(sex_key)
    train_df = build_training_df(df_sex, use_vars)
    if len(train_df) < 4:
        raise ValueError(f"Too few rows for training after filtering by sex={sex_key} and predictors={use_vars}")

    # CV
    cv = cv_run(train_df, use_vars, method=method, k=k, seed=seed)

    # Final fit
    X = train_df[use_vars]
    y = train_df["Sum_Verts"].values

    if method == "ols":
        X_ = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, X_).fit()
        x_user = pd.DataFrame([user_values])[use_vars]
        x_user_ = sm.add_constant(x_user, has_constant="add")
        pred = float(fit.get_prediction(x_user_).predicted_mean[0])

        # 95% PI (obs)
        pred_res = fit.get_prediction(x_user_).summary_frame(alpha=0.05)
        pi_lower = float(pred_res["obs_ci_lower"].iloc[0])
        pi_upper = float(pred_res["obs_ci_upper"].iloc[0])

    else:
        rf = RandomForestRegressor(n_estimators=500, random_state=seed)
        rf.fit(X, y)
        x_user = pd.DataFrame([user_values])[use_vars]
        pred = float(rf.predict(x_user)[0])
        tr_preds = rf.predict(X)
        se = float(np.std(tr_preds, ddof=1) / np.sqrt(len(tr_preds)))
        tcrit = 1.96
        pi_lower = pred - tcrit * se
        pi_upper = pred + tcrit * se
        fit = rf

    return {
        "predictors": use_vars,
        "prediction": pred,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "cv": cv,
        "model": fit,
        "train_n": len(train_df)
    }

def random_feature_benchmark_fixed(user_values: dict, sex_key: str, method: str = "ols",
                                   trials: int = 1000, k: int = 5, seed: int = 2025):
    """p = #user predictors. For each trial, sample p random features, run CV, collect R2/RMSE."""
    p = len([v for v in user_values if v in VERTE_NAMES and float(user_values[v]) != 0])
    if p < 1:
        return {"summary": {}, "details": pd.DataFrame()}

    df_sex = get_sex_filtered_df(sex_key)
    all_vars = [v for v in VERTE_NAMES if v in df_sex.columns]

    r2_means, rmse_means = [], []
    rng = random.Random(seed)

    for t in range(trials):
        feats = rng.sample(all_vars, k=p)
        train_df = build_training_df(df_sex, feats)
        if len(train_df) < 4:
            r2_means.append(np.nan); rmse_means.append(np.nan); continue
        cv = cv_run(train_df, feats, method=method, k=k, seed=seed + t)
        r2_means.append(cv["summary"]["R2_mean"])
        rmse_means.append(cv["summary"]["RMSE_mean"])

    details = pd.DataFrame({"trial": np.arange(1, trials+1),
                            "R2_mean": r2_means,
                            "RMSE_mean": rmse_means})
    summary = {
        "p": p,
        "trials": trials,
        "valid_trials": int(np.sum(np.isfinite(details["R2_mean"]) & np.isfinite(details["RMSE_mean"]))),
        "R2_avg": float(np.nanmean(details["R2_mean"])),
        "R2_sd": float(np.nanstd(details["R2_mean"], ddof=1)),
        "RMSE_avg": float(np.nanmean(details["RMSE_mean"])),
        "RMSE_sd": float(np.nanstd(details["RMSE_mean"], ddof=1)),
    }
    return {"summary": summary, "details": details}

# -----------------------------
# Plot helpers
# -----------------------------
def save_cv_boxplots(cv_details: list, title_prefix: str = "CV"):
    if not cv_details:
        return None
    r2_vals = [d["R2"] for d in cv_details if d["R2"] is not None]
    rmse_vals = [d["RMSE"] for d in cv_details if d["RMSE"] is not None]
    if len(r2_vals) == 0 or len(rmse_vals) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].boxplot(r2_vals, vert=True)
    axes[0].set_title(f"{title_prefix} R²"); axes[0].set_ylabel("R²")

    axes[1].boxplot(rmse_vals, vert=True)
    axes[1].set_title(f"{title_prefix} RMSE"); axes[1].set_ylabel("RMSE")

    fname = f"cv_{uuid.uuid4().hex}.png"
    path = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return url_for("static", filename=f"plots/{fname}")

def save_benchmark_histograms(bench_df: pd.DataFrame):
    if bench_df is None or bench_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # RMSE
    rmse_vals = bench_df["RMSE_mean"].dropna().values
    axes[0].hist(rmse_vals, bins=30, edgecolor="black")
    axes[0].axvline(np.mean(rmse_vals), linestyle="--", color="red", linewidth=1.5)
    axes[0].set_title("RMSE distribution (random-feature 1000 trials)")
    axes[0].set_xlabel("RMSE"); axes[0].set_ylabel("Count")

    # R2
    r2_vals = bench_df["R2_mean"].dropna().values
    axes[1].hist(r2_vals, bins=30, edgecolor="black")
    axes[1].axvline(np.mean(r2_vals), linestyle="--", color="red", linewidth=1.5)
    axes[1].set_title("R² distribution (random-feature 1000 trials)")
    axes[1].set_xlabel("R²"); axes[1].set_ylabel("Count")

    fname = f"bench_{uuid.uuid4().hex}.png"
    path = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return url_for("static", filename=f"plots/{fname}")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def predict():
    cervical = ["C2", "C3", "C4", "C5", "C6", "C7"]
    thoracic = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
    lumbar = ["L1", "L2", "L3", "L4", "L5"]
    return render_template("predict.html", cervical=cervical, thoracic=thoracic, lumbar=lumbar)

@app.route("/about")
def about():
    return render_template("about.html")

# simple in-memory store for one hop
_USER_STORE = {}

@app.route("/input", methods=["POST"])
def input():
    # gather up to 23 vertebrae
    vals = []
    for i in range(1, 24):
        v = request.form.get(f"Vertebrae{i}", "").strip()
        vals.append(v)

    # map to dict with names
    user_values = {}
    for idx, name in enumerate(VERTE_NAMES):
        try:
            val = float(vals[idx]) if vals[idx] != "" else 0.0
        except:
            val = 0.0
        if val != 0.0:
            user_values[name] = val

    sex_form = request.form.get("Sex", "Unknown")
    sex_key = sex_key_from_form(sex_form)

    reg_form = request.form.get("Reg", "Linear")
    method = "ols" if reg_form.lower().startswith("linear") else "rf"

    token = uuid.uuid4().hex
    _USER_STORE[token] = {"user_values": user_values, "sex_key": sex_key, "method": method}
    return redirect(url_for("getPrediction", token=token))

@app.route("/results", methods=["GET"])
def getPrediction():
    token = request.args.get("token")
    if not token or token not in _USER_STORE:
        return "Invalid request.", 400

    user_blob = _USER_STORE.pop(token)
    user_values = user_blob["user_values"]
    sex_key = user_blob["sex_key"]
    method = user_blob["method"]

    if len(user_values) == 0:
        return render_template("results.html")  # 메시지 없이

    # Fit + Predict with ONLY user-provided features
    try:
        res = fit_and_predict(user_values, sex_key=sex_key, method=method, k=5, seed=42)
    except Exception as e:
        return render_template("results.html")  # 메시지 없이

    # CV plot
    cv_img = save_cv_boxplots(res["cv"]["details"], title_prefix=f"{method.upper()} ({sex_key})")

    # Random-feature benchmark (fixed 1000 trials)
    bench = random_feature_benchmark_fixed(user_values, sex_key=sex_key, method=method,
                                           trials=1000, k=5, seed=2025)
    bench_img = save_benchmark_histograms(bench["details"])

    # benchmark summary
    bsum = bench.get("summary", {}) or {}
    r2_bench_avg  = bsum.get("R2_avg")
    r2_bench_sd   = bsum.get("R2_sd")
    rmse_bench_avg = bsum.get("RMSE_avg")
    rmse_bench_sd  = bsum.get("RMSE_sd")
    bench_trials   = bsum.get("trials")
    bench_p        = bsum.get("p")
    bench_valid    = bsum.get("valid_trials")

    r2_mean = res["cv"]["summary"]["R2_mean"]
    rmse_mean = res["cv"]["summary"]["RMSE_mean"]

    return render_template(
        "results.html",
        # core outputs
        prediction=round(res["prediction"], 1),
        pi_lower=round(res["pi_lower"], 1),
        pi_upper=round(res["pi_upper"], 1),
        predictors=", ".join(res["predictors"]),
        sex=sex_key,
        method=method.upper(),
        train_n=res["train_n"],
        cv_k=res["cv"]["summary"]["k"],
        cv_r2_mean=None if r2_mean is None else round(r2_mean, 4),
        cv_rmse_mean=None if rmse_mean is None else round(rmse_mean, 4),
        # images
        cv_img=cv_img,
        bench_img=bench_img,
        # benchmark summary
        bench_p=bench_p,
        bench_trials=bench_trials,
        bench_valid=bench_valid,
        r2_bench_avg=r2_bench_avg,
        r2_bench_sd=r2_bench_sd,
        rmse_bench_avg=rmse_bench_avg,
        rmse_bench_sd=rmse_bench_sd
    )

if __name__ == '__main__':
    app.run(debug=True)
