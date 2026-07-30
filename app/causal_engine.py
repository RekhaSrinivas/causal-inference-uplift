"""
Causal inference pipeline on the full Criteo dataset (~14M rows):
  - Propensity score estimation (logistic regression)
  - 1:1 nearest-neighbour matching with caliper
  - ATE with confidence interval
  - T-Learner uplift model (HistGBM — fast on large datasets)
  - Covariate balance checks (SMD)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, qini_curve, perfect_qini_curve
from scipy import stats

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOME = "visit"


def _smd(df, features, T):
    """Standardised mean difference for each feature."""
    result = {}
    for col in features:
        if col not in df.columns:
            continue
        x1 = df.loc[T == 1, col].values.astype(float)
        x0 = df.loc[T == 0, col].values.astype(float)
        pooled_std = np.sqrt((np.var(x1) + np.var(x0)) / 2)
        if pooled_std == 0:
            result[col] = 0.0
        else:
            result[col] = round(abs(np.mean(x1) - np.mean(x0)) / pooled_std, 4)
    return result


def compute_propensity_scores(df):
    print("[engine] Computing propensity scores...")
    X = df[FEATURES].values
    T = df[TREATMENT].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # solver='saga' handles large datasets better
    model = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
    model.fit(X_scaled, T)
    ps = model.predict_proba(X_scaled)[:, 1]

    return {
        "ps": ps,
        "T": T,
        "scaler": scaler,
        "model": model,
        "smd_before": _smd(df, FEATURES, T),
    }


def propensity_score_matching(df, ps_result):
    """
    1:1 matching on propensity score. With 14M rows we match each control
    unit to its nearest treated unit (since control is the smaller group).
    """
    print("[engine] Running propensity score matching...")
    ps = ps_result["ps"]
    T = ps_result["T"]

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    # For each treated unit, find the nearest control on propensity score
    # (matching with replacement — controls can be reused since the control
    # pool is much smaller than treated in the Criteo trial).
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    matched_control = control_idx[indices.flatten()]
    within_caliper = distances.flatten() < 0.05

    mt = treated_idx[within_caliper]
    mc = matched_control[within_caliper]

    matched_df = pd.concat([df.iloc[mt], df.iloc[mc]]).reset_index(drop=True)
    T_matched = matched_df[TREATMENT].values
    smd_after = _smd(matched_df, FEATURES, T_matched)

    y1 = matched_df.loc[matched_df[TREATMENT] == 1, OUTCOME].values
    y0 = matched_df.loc[matched_df[TREATMENT] == 0, OUTCOME].values

    ate = float(np.mean(y1) - np.mean(y0))
    se = float(np.sqrt(np.var(y1) / len(y1) + np.var(y0) / len(y0)))
    _, p_value = stats.ttest_ind(y1, y0)

    print(f"[engine] PSM done: {mt.shape[0]:,} matched, ATE={ate*100:+.2f}pp")

    return {
        "n_matched": int(mt.shape[0]),
        "n_dropped": int(np.sum(~within_caliper)),
        "ate": round(ate, 4),
        "ate_pct": round(ate * 100, 2),
        "ci_low": round(ate - 1.96 * se, 4),
        "ci_high": round(ate + 1.96 * se, 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "control_conversion_rate": round(float(np.mean(y0)), 4),
        "treated_conversion_rate": round(float(np.mean(y1)), 4),
        "smd_before": ps_result["smd_before"],
        "smd_after": smd_after,
    }


def uplift_modeling(df):
    """
    T-Learner using HistGradientBoosting. We do a train/test split so the
    Qini evaluation is honest (out-of-sample). 80% train, 20% test, stratified
    on treatment so both sets have the same treated/control ratio.
    """
    print("[engine] Training T-Learner uplift model...")
    X = df[FEATURES].values
    T = df[TREATMENT].values
    Y = df[OUTCOME].values

    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
        X, T, Y, test_size=0.2, random_state=42, stratify=T
    )

    params = dict(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42)

    m_treat = HistGradientBoostingClassifier(**params)
    m_ctrl = HistGradientBoostingClassifier(**params)
    m_treat.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    m_ctrl.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])

    # predict uplift on test set
    uplift_te = m_treat.predict_proba(X_te)[:, 1] - m_ctrl.predict_proba(X_te)[:, 1]

    # split into quartiles based on test-set predictions
    q25, q50, q75 = np.percentile(uplift_te, [25, 50, 75])
    segments = np.digitize(uplift_te, [q25, q50, q75])

    segment_stats = []
    for q in range(4):
        mask = segments == q
        t_mask = mask & (T_te == 1)
        c_mask = mask & (T_te == 0)
        segment_stats.append({
            "quartile": f"Q{q + 1}",
            "n": int(np.sum(mask)),
            "avg_uplift_pct": round(float(np.mean(uplift_te[mask])) * 100, 2),
            "treatment_conversion": round(float(np.mean(Y_te[t_mask])) if t_mask.any() else 0, 4),
            "control_conversion": round(float(np.mean(Y_te[c_mask])) if c_mask.any() else 0, 4),
            "recommendation": _segment_rec(q),
        })

    # Qini evaluation on test set
    print("[engine] Computing Qini curve...")
    qini_score = float(qini_auc_score(y_true=Y_te, uplift=uplift_te, treatment=T_te))
    x_model, y_model = qini_curve(y_true=Y_te, uplift=uplift_te, treatment=T_te)
    x_perfect, y_perfect = perfect_qini_curve(y_true=Y_te, treatment=T_te)

    # downsample curves to ~100 points for the frontend
    def _downsample(x, y, n=100):
        if len(x) <= n:
            return x.tolist(), y.tolist()
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        return x[idx].tolist(), y[idx].tolist()

    xm, ym = _downsample(np.asarray(x_model), np.asarray(y_model))
    xp, yp = _downsample(np.asarray(x_perfect), np.asarray(y_perfect))

    print(f"[engine] Uplift model done. Qini score: {qini_score:.4f}")

    return {
        "uplift_scores": uplift_te.tolist(),
        "mean_uplift_pct": round(float(np.mean(uplift_te)) * 100, 2),
        "segment_stats": segment_stats,
        "qini": {
            "score": round(qini_score, 4),
            "x_model": xm,
            "y_model": ym,
            "x_perfect": xp,
            "y_perfect": yp,
        },
    }


def _segment_rec(q):
    return [
        "Do not target — negligible or negative uplift",
        "Low priority — modest response",
        "Good target — above-average response",
        "Priority target — highest uplift segment",
    ][q]


def run_full_analysis(df):
    T = df[TREATMENT].values
    Y = df[OUTCOME].values

    overview = {
        "n_total": len(df),
        "n_treated": int(np.sum(T)),
        "n_control": int(np.sum(T == 0)),
        "treatment_rate": round(float(np.mean(T)), 3),
        "overall_conversion": round(float(np.mean(Y)), 4),
        "naive_ate_pct": round(float(np.mean(Y[T == 1]) - np.mean(Y[T == 0])) * 100, 2),
        "dataset": "Criteo Uplift Modeling Dataset (13.9M rows)",
        "experiment": "Online advertising incrementality trial (ad bid vs no bid)",
    }

    ps_result = compute_propensity_scores(df)
    psm_result = propensity_score_matching(df, ps_result)
    uplift_result = uplift_modeling(df)

    # subsample for JSON response (don't send 14M points to the browser)
    import random
    random.seed(42)
    ps = ps_result["ps"].tolist()
    ps_t = [ps[i] for i, t in enumerate(T) if t == 1]
    ps_c = [ps[i] for i, t in enumerate(T) if t == 0]
    ps_t_sample = random.sample(ps_t, min(800, len(ps_t)))
    ps_c_sample = random.sample(ps_c, min(800, len(ps_c)))
    uplift_sample = random.sample(uplift_result["uplift_scores"],
                                  min(2000, len(uplift_result["uplift_scores"])))

    return {
        "overview": overview,
        "ps_treated": ps_t_sample,
        "ps_control": ps_c_sample,
        "psm": psm_result,
        "uplift": {**uplift_result, "uplift_scores": uplift_sample},
    }
