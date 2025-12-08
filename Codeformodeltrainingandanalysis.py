
# -*- coding: utf-8 -*-
"""
Master script: compute & export ALL indicators in one go
--------------------------------------------------------
Outputs (under OUTPUT_DIR):
  CSVs
  - train_actual_vs_pred.csv
  - test_actual_vs_pred.csv
  - oob_actual_vs_pred.csv
  - feature_importance_detailed.csv        # Node purity (MDI), Permutation MSE Increase (+ std, p-value)
  - mmd_unconditional.csv                  # MMD, NOT, NON, NORT, NPI_sum, p-value, NPI_norm
  - cond_mmd_all_pairs.csv                 # A:B conditional MMD + frequency + B's unconditional MMD
  - top30_interactions.csv
  - indicator_scores.csv                   # raw scores for MMD/NON/NORT/NPI/InMSE
  - indicator_ranks.csv                    # ranks derived from the above scores

  Figures
  - supp_fig11.png / supp_fig11.pdf        # Top-30 conditional MMD plot (A:B interactions)
  - Supplementary_Fig10_replica.tif        # Triangle correlation among indicator ranks (MMD/NON/NORT/NPI/InMSE)
  - scatter_train_test.png                 # quick QA: actual vs predicted (train/test)

Notes
- Requires: numpy, pandas, scikit-learn, scipy, matplotlib, joblib (optional), statsmodels (optional for CI)
- Designed to be robust to CSV encodings; adjust DATA_CSV/OUTPUT_DIR as needed.
"""

import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------- User CONFIG ----------------------
DATA_CSV   = "Dataset.csv"                 # <- change if needed
OUTPUT_DIR = Path("outputs_all")      # <- change if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature / target config (align with your dataset columns)
FEATURES = [
    'Crop_TypeMaize', 'Crop_TypeRice', 'Crop_TypeWheat',
    'Tmp', 'Prec', 'BD', 'Clay', 'SOC', 'pH',
    'FertilizerAN', 'FertilizerMA', 'FertilizerO', 'FertilizerU', 'FertilizerEEF',
    'Tillage practice', 'Irrigation', 'Fertilizer placement', 'Nrate'
]
TARGET = 'EF'

# Modeling config
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Permutation importance
PERM_REPEATS = 30

# Interaction & plotting config
TOP_A_GIVEN       = None         # e.g., ["Tmp","Nrate","CEC","Water input"]
TOP_A_K           = 10           # auto-pick top-K A features
TOPN_INTERACTIONS = 30
SAVE_PNG = True
SAVE_PDF = True

# ---------------------------------------------------------

def read_csv_safely(path):
    """
    Try several encodings before giving up.
    """
    tries = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
    last_err = None
    for enc in tries:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

print(f"[INFO] Reading data: {DATA_CSV}")
data = read_csv_safely(DATA_CSV)

# Validate columns
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns in data: {missing}")

X = data[FEATURES].copy()
y = data[TARGET].copy()

# ---------------------- Train / Test split ----------------------
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------------------- GridSearch & Fit ----------------------
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_features': ['sqrt', 'log2', 1.0],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}
base = RandomForestRegressor(random_state=RANDOM_STATE)
gs = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    cv=10,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error'
)
print("[INFO] Running GridSearchCV ...")
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
print("[OK] Best hyperparameters:", gs.best_params_)

# CV on train only
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
print(f"[Train-CV] MSE={-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

# Fit final model on train
final_model = RandomForestRegressor(**gs.best_params_, random_state=RANDOM_STATE)
final_model.fit(X_train, y_train)

# ---------------------- Predictions & export ----------------------
from sklearn.metrics import mean_squared_error, r2_score

y_pred_train = final_model.predict(X_train)
y_pred_test  = final_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = train_mse ** 0.5
train_r2  = r2_score(y_train, y_pred_train)

test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = test_mse ** 0.5
test_r2  = r2_score(y_test, y_pred_test)

print(f"[Train] RMSE={train_rmse:.6f}, R2={train_r2:.4f}")
print(f"[Test ] RMSE={test_rmse:.6f}, R2={test_r2:.4f}")

train_out = pd.DataFrame({
    'index': X_train.index,
    'Actual': y_train.values,
    'Predicted': y_pred_train,
    'Residual': y_train.values - y_pred_train
})
test_out = pd.DataFrame({
    'index': X_test.index,
    'Actual': y_test.values,
    'Predicted': y_pred_test,
    'Residual': y_test.values - y_pred_test
})
train_out.to_csv(OUTPUT_DIR / "train_actual_vs_pred.csv", index=False, encoding='utf-8-sig')
test_out.to_csv(OUTPUT_DIR / "test_actual_vs_pred.csv",  index=False, encoding='utf-8-sig')

# ---------------------- OOB evaluation ----------------------
oob_params = {k: v for k, v in final_model.get_params().items() if k in RandomForestRegressor().get_params()}
oob_params.update(dict(bootstrap=True, oob_score=True, random_state=RANDOM_STATE))
oob_model = RandomForestRegressor(**oob_params)
oob_model.fit(X_train, y_train)
y_pred_oob = oob_model.oob_prediction_
oob_mse = mean_squared_error(y_train, y_pred_oob)
oob_rmse = oob_mse ** 0.5
oob_r2 = oob_model.oob_score_
print(f"[OOB  ] RMSE={oob_rmse:.6f}, R2={oob_r2:.4f}")

oob_out = pd.DataFrame({
    'index': X_train.index,
    'Actual': y_train.values,
    'Predicted': y_pred_oob,
    'Residual': y_train.values - y_pred_oob
})
oob_out.to_csv(OUTPUT_DIR / "oob_actual_vs_pred.csv", index=False, encoding='utf-8-sig')

# ---------------------- Feature importances ----------------------
# 1) Node purity increase (MDI)
node_purity = final_model.feature_importances_

# 2) Permutation importance on test set
from sklearn.inspection import permutation_importance
perm = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
    scoring='neg_mean_squared_error'
)
mse_increase = perm.importances_mean
mse_std      = perm.importances_std

# 3) One-sided binomial p-value for proportion of >0 improvements
try:
    from scipy.stats import binomtest as _binomtest
    def one_sided_p(n_pos, n_total):
        return _binomtest(n_pos, n_total, 0.5, alternative='greater').pvalue
except Exception:
    from scipy.stats import binom_test as _binomtest
    def one_sided_p(n_pos, n_total):
        return _binomtest(n_pos, n_total, 0.5, alternative='greater')

imps = perm.importances
if imps.shape[0] != len(FEATURES) and imps.shape[1] == len(FEATURES):
    imps = imps.T
p_values = [one_sided_p(int((row > 0).sum()), int(row.size)) for row in imps]

imp_df = pd.DataFrame({
    'Feature': FEATURES,
    'Node_Purity_Increase': node_purity,
    'MSE_Increase': mse_increase,
    'MSE_Std': mse_std,
    'P_value': p_values
}).sort_values(by='MSE_Increase', ascending=False)
imp_df.to_csv(OUTPUT_DIR / "feature_importance_detailed.csv", index=False, encoding='utf-8-sig')

# ---------------------- Unconditional MMD / NOT / NON / NORT / NPI ----------------------
def compute_tree_depths(tree):
    cl = tree.children_left
    cr = tree.children_right
    node_count = tree.node_count
    depth = np.zeros(node_count, dtype=np.int32)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        left = cl[node]
        if left != -1:
            stack.append((left, d + 1))
            stack.append((cr[node], d + 1))
    return depth

name_by_idx = {i: n for i, n in enumerate(FEATURES)}
min_depths = {n: [] for n in FEATURES}
NOT = {n: 0 for n in FEATURES}
NON = {n: 0 for n in FEATURES}
NORT = {n: 0 for n in FEATURES}
NPI_sum = {n: 0.0 for n in FEATURES}

for est in final_model.estimators_:
    tree = est.tree_
    depth = compute_tree_depths(tree)
    cl, cr, feat = tree.children_left, tree.children_right, tree.feature
    impurity = tree.impurity
    w = tree.weighted_n_node_samples

    per_tree_min = {n: math.inf for n in FEATURES}
    used = set()

    for node in range(tree.node_count):
        fidx = feat[node]
        if fidx >= 0:
            fname = name_by_idx.get(fidx)
            if fname is None: 
                continue
            d = int(depth[node])
            NON[fname] += 1
            used.add(fname)
            per_tree_min[fname] = min(per_tree_min[fname], d)

            l, r = cl[node], cr[node]
            wp = w[node]
            if l != -1 and r != -1 and wp > 0:
                wl, wr = w[l], w[r]
                child_imp = (wl * impurity[l] + wr * impurity[r]) / wp
                dec = max(impurity[node] - child_imp, 0.0)
                NPI_sum[fname] += dec

    for fname in FEATURES:
        if fname in used:
            NOT[fname] += 1
            if per_tree_min[fname] == 0:
                NORT[fname] += 1
            min_depths[fname].append(per_tree_min[fname])
        else:
            min_depths[fname].append(np.nan)

rows = []
for fname in FEATURES:
    arr = np.array(min_depths[fname], dtype=float)
    mmd = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan
    rows.append({
        "feature": fname,
        "MMD": mmd,
        "NOT": int(NOT[fname]),
        "NON": int(NON[fname]),
        "NORT": int(NORT[fname]),
        "NPI_sum": float(NPI_sum[fname]),
    })
mmd_df = pd.DataFrame(rows)

# Binomial test for usage enrichment at split nodes
total_nodes_with_splits = int(mmd_df["NON"].sum())
p0 = 1.0 / len(FEATURES)
from scipy.stats import binomtest
mmd_df["p_value"] = mmd_df["NON"].apply(lambda k: binomtest(int(k), total_nodes_with_splits, p0, alternative="greater").pvalue if total_nodes_with_splits>0 else np.nan)
mmd_df["NPI_norm"] = mmd_df["NPI_sum"] / mmd_df["NPI_sum"].max() if mmd_df["NPI_sum"].max() > 0 else 0.0
mmd_df.sort_values(["MMD","NOT"], ascending=[True, False]).to_csv(OUTPUT_DIR / "mmd_unconditional.csv", index=False)

# ---------------------- Conditional MMD: A:B ----------------------
def collect_conditional_mmd(model, feat_cols, topA, mmd_uncond):
    name_by_idx = {i: n for i, n in enumerate(feat_cols)}
    results = []
    for A in topA:
        for est in model.estimators_:
            tree = est.tree_
            cl, cr, feat = tree.children_left, tree.children_right, tree.feature
            depth = compute_tree_depths(tree)

            A_nodes = [i for i in range(tree.node_count) if feat[i] >= 0 and name_by_idx[feat[i]] == A]
            if not A_nodes:
                continue
            root_A = min(A_nodes, key=lambda i: depth[i])

            stack = [root_A]
            sub_depths = {}
            while stack:
                node = stack.pop()
                fidx = feat[node]
                if fidx >= 0:
                    fname = name_by_idx[fidx]
                    rel_depth = depth[node] - depth[root_A]
                    if fname != A:
                        if fname not in sub_depths or rel_depth < sub_depths[fname]:
                            sub_depths[fname] = rel_depth
                    if cl[node] != -1: stack.append(cl[node])
                    if cr[node] != -1: stack.append(cr[node])
            for B, d in sub_depths.items():
                results.append((A, B, d))
    cond_df = pd.DataFrame(results, columns=["A","B","cond_MMD"])
    if len(cond_df)==0:
        return cond_df
    cond_summary = cond_df.groupby(["A","B"]).agg(
        cond_MMD=("cond_MMD","mean"),
        count=("cond_MMD","size")
    ).reset_index()
    cond_summary = cond_summary.merge(
        mmd_uncond[["feature","MMD"]],
        left_on="B", right_on="feature", how="left"
    ).rename(columns={"MMD":"MMD_B_uncond"}).drop(columns="feature")
    return cond_summary

# Choose A set
if TOP_A_GIVEN:
    topA = [f for f in TOP_A_GIVEN if f in FEATURES]
else:
    tmp = mmd_df.copy()
    tmp["sig"] = (tmp["p_value"] < 0.01).astype(int)
    tmp = tmp.sort_values(["sig","NPI_norm","MMD"], ascending=[False, False, True])
    topA = list(tmp["feature"].head(TOP_A_K))

cond_df = collect_conditional_mmd(final_model, FEATURES, topA, mmd_df)
cond_df.to_csv(OUTPUT_DIR / "cond_mmd_all_pairs.csv", index=False, encoding='utf-8-sig')

top30 = cond_df.sort_values("count", ascending=False).head(TOPN_INTERACTIONS) if len(cond_df)>0 else pd.DataFrame(columns=["A","B","cond_MMD","count","MMD_B_uncond"])
top30.to_csv(OUTPUT_DIR / "top30_interactions.csv", index=False, encoding='utf-8-sig')

# ---------------------- Plot: Top-30 interactions ----------------------
import matplotlib
import matplotlib.pyplot as plt

if len(top30) > 0:
    plt.figure(figsize=(12,6))
    top30 = top30.sort_values("count", ascending=False).reset_index(drop=True)
    xpos = np.arange(len(top30))
    sc = plt.scatter(xpos, top30["cond_MMD"].values, c=top30["count"].values, s=80, edgecolor="k")
    plt.scatter(xpos, top30["MMD_B_uncond"].values, c="black", s=40, marker="o")
    plt.xticks(xpos, [f"{a}:{b}" for a,b in zip(top30["A"], top30["B"])], rotation=90, fontsize=9)
    plt.ylabel("Conditional MMD (B | A subtree)")
    plt.xlabel("Top 30 A:B interactions (sorted by frequency)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Interaction count (frequency)")
    plt.title("Feature Interactions (Top 30)")
    plt.tight_layout()
    if SAVE_PNG: plt.savefig(OUTPUT_DIR / "supp_fig11.png", dpi=300)
    if SAVE_PDF: plt.savefig(OUTPUT_DIR / "supp_fig11.pdf")
    plt.close()

# ---------------------- Build indicator scores & ranks (for triangle plot) ----------------------
# Scores
scores_df = pd.DataFrame({
    "Feature": FEATURES
})
scores_df = scores_df.merge(
    mmd_df[["feature","MMD","NON","NORT","NPI_sum"]].rename(columns={"feature":"Feature"}),
    on="Feature", how="left"
)
scores_df = scores_df.merge(
    imp_df[["Feature","MSE_Increase"]].rename(columns={"MSE_Increase":"InMSE"}),
    on="Feature", how="left"
)
scores_df.to_csv(OUTPUT_DIR / "indicator_scores.csv", index=False, encoding='utf-8-sig')

# Ranks: lower MMD is better (ascending=True), the others descending
ranks_df = pd.DataFrame({"Feature": FEATURES})
ranks_df["rank_MMD"]  = scores_df["MMD"].rank(ascending=True,  method="min")
ranks_df["rank_NON"]  = scores_df["NON"].rank(ascending=False, method="min")
ranks_df["rank_NORT"] = scores_df["NORT"].rank(ascending=False, method="min")
ranks_df["rank_NPI"]  = scores_df["NPI_sum"].rank(ascending=False, method="min")
ranks_df["rank_InMSE"]= scores_df["InMSE"].rank(ascending=False, method="min")
ranks_df.to_csv(OUTPUT_DIR / "indicator_ranks.csv", index=False, encoding='utf-8-sig')

# ---------------------- Triangle correlation plot (5x5) ----------------------
from scipy.stats import pearsonr, gaussian_kde

INDICATOR_ORDER = ['MMD', 'NON', 'NORT', 'NPI', 'InMSE']
ALIASES = {
    'MMD'   : ['rank_MMD', 'MMD'],
    'NON'   : ['rank_NON', 'NON'],
    'NORT'  : ['rank_NORT', 'NORT'],
    'NPI'   : ['rank_NPI', 'NPI_sum'],
    'InMSE' : ['rank_InMSE', 'InMSE']
}

def build_rank_matrix(ranks_df, scores_df):
    out = pd.DataFrame(index=ranks_df.index)
    for std_name, alias_list in ALIASES.items():
        # Prefer rank columns if present; otherwise rank scores on the fly
        rank_col = next((a for a in alias_list if a in ranks_df.columns and a.lower().startswith("rank")), None)
        if rank_col is not None:
            out[std_name] = pd.to_numeric(ranks_df[rank_col], errors="coerce")
            continue
        score_col = next((a for a in alias_list if a in scores_df.columns and not a.lower().startswith("rank")), None)
        if score_col is not None:
            series = pd.to_numeric(scores_df[score_col], errors="coerce")
            if std_name == "MMD":
                out[std_name] = series.rank(ascending=True, method="min")
            else:
                out[std_name] = series.rank(ascending=False, method="min")
    out = out[[c for c in INDICATOR_ORDER if c in out.columns]].dropna(how="all")
    return out

ranks_plot = build_rank_matrix(ranks_df, scores_df)

def significance_stars(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return 'ns'

def draw_triangle_plot(ranks: pd.DataFrame, out_path: Path):
    cols = list(ranks.columns)
    m = len(cols)
    cell = 2.8
    fig, axes = plt.subplots(m, m, figsize=(cell*m, cell*m), constrained_layout=True)

    if m == 1:
        axes = np.array([[axes]])

    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            xi = cols[j]
            yi = cols[i]

            ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)

            if i == j:
                data = pd.to_numeric(ranks[xi], errors='coerce').dropna()
                if len(data) > 1:
                    try:
                        kde = gaussian_kde(data)
                        xs = np.linspace(data.min(), data.max(), 200)
                        ax.plot(xs, kde(xs), linewidth=1.8)
                    except Exception:
                        bins = min(max(len(pd.unique(data)), 6), 20)
                        ax.hist(data, bins=bins, alpha=0.85)
                else:
                    ax.plot([0,1],[0,0])
            elif i > j:
                x = pd.to_numeric(ranks[xi], errors='coerce')
                y = pd.to_numeric(ranks[yi], errors='coerce')
                mask = x.notna() & y.notna()
                x = x[mask]; y = y[mask]
                ax.scatter(x, y, s=14, alpha=0.55)
                if len(x) >= 5:
                    # simple quadratic fit
                    coef = np.polyfit(x, y, 2)
                    grid = np.linspace(x.min(), x.max(), 120)
                    fit_y = np.polyval(coef, grid)
                    ax.plot(grid, fit_y, linewidth=1.6)
            else:
                x = pd.to_numeric(ranks[xi], errors='coerce')
                y = pd.to_numeric(ranks[yi], errors='coerce')
                mask = x.notna() & y.notna()
                x = x[mask]; y = y[mask]
                if len(x) >= 3:
                    r, p = pearsonr(x, y)
                    txt = f"r = {r:.2f}\n{significance_stars(p)}"
                else:
                    txt = "n/a"
                ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=16)
                for s in ax.spines.values():
                    s.set_visible(False)
            ax.tick_params(labelbottom=False, labelleft=False)
            if i == m - 1:
                ax.tick_params(labelbottom=True, labelsize=10)
            if j == 0:
                ax.tick_params(labelleft=True, labelsize=10)

    for j, name in enumerate(cols):
        axes[0, j].set_title(name, fontsize=16, pad=8)
    for i, name in enumerate(cols):
        ax_right = axes[i, -1]
        ax_right.set_ylabel(name, fontsize=16, rotation=270, labelpad=16)
        ax_right.yaxis.set_label_position('right')
        ax_right.yaxis.tick_right()

    fig.suptitle('Relations between rankings according to different measures',
                 fontname='Arial', fontsize=18, fontweight='bold', y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

draw_triangle_plot(ranks_plot, OUTPUT_DIR / "Supplementary_Fig10_replica.tif")

# ---------------------- Quick QA scatter (train/test) ----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_train, y_pred_train, s=12)
mn, mx = min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())
plt.plot([mn,mx],[mn,mx],'k--',linewidth=1)
plt.title(f"Train R^2={train_r2:.3f}, RMSE={train_rmse:.3f}")
plt.xlabel("Actual"); plt.ylabel("Predicted")

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_test, s=12)
mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
plt.plot([mn,mx],[mn,mx],'k--',linewidth=1)
plt.title(f"Test R^2={test_r2:.3f}, RMSE={test_rmse:.3f}")
plt.xlabel("Actual"); plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_train_test.png", dpi=300)
plt.close()

print("[DONE] All artifacts written to:", OUTPUT_DIR)
