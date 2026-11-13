"""Game-median-normalized playtime ALS, 3 weighting modes x 3 betas.

Runs on existing Steam train/validation and PS test splits.
Mode A: below-median penalty including zero playtime.
Mode B: above-median bonus only.
Mode C: below-median penalty for positive playtime, but zero stays neutral.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares
from als_playtime import (
    build_validation_matrix, evaluate_all_users,
    validate_interactions, validate_train_playtime
)
from als_ps_test import build_ps_matrix, evaluate_als

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
BETAS = (0.25, 0.5, 1.0)
MODES = ("A", "B", "C")
METRICS = ("recall_at_10", "ndcg_at_10", "hit_at_10")
ALPHA = 2.0
SEED = 42


def data_frame(name, columns):
    return pd.read_parquet(DATA / (name + ".parquet"), columns=columns)


def prepare():
    train = data_frame("steam_train",
                       ["user_id", "unified_game_id", "playtime_forever"])
    validate_interactions(train, "Steam Train")
    train = validate_train_playtime(train)
    u, users = pd.factorize(train["user_id"], sort=True)
    i, games = pd.factorize(train["unified_game_id"], sort=True)
    game_to_idx = {game: idx for idx, game in enumerate(games)}
    positive = train[train["playtime_forever"] > 0]
    med_stats = (positive.groupby("unified_game_id")["playtime_forever"]
                 .agg(["median", "count"]))
    eligible_medians = med_stats.loc[med_stats["count"] >= 20, "median"] / 60
    fallback = float(positive["playtime_forever"].median() / 60)
    med = (train["unified_game_id"].map(eligible_medians)
           .fillna(fallback).to_numpy(dtype=np.float64))
    hours = train["playtime_forever"].to_numpy(dtype=np.float64) / 60
    cap = float(np.quantile(hours[hours > 0], 0.99))
    ratios = (np.minimum(hours, cap) + 1.0) / (np.minimum(med, cap) + 1.0)
    print(f"Training: {len(train):,} interactions, {len(users):,} users, {len(games):,} games", flush=True)
    print(f"Positive-playtime median fallback: {fallback:.4f} h; P99 cap: {cap:.4f} h", flush=True)
    print(f"Games with >=20 positive records: {len(eligible_medians):,}", flush=True)
    return dict(u=u, i=i, shape=(len(users), len(games)),
                game_ids=games, game_to_idx=game_to_idx,
                ratios=ratios, zero=hours == 0)


def train_model(prep, mode, beta):
    w = np.clip(np.power(prep["ratios"], beta), 0.5, 2.0)
    if mode == "B":
        w = np.maximum(w, 1.0)
    elif mode == "C":
        w[prep["zero"]] = 1.0
    elif mode != "A":
        raise ValueError(mode)
    confidence = (1.0 + ALPHA * w).astype(np.float32)
    matrix = csr_matrix((confidence, (prep["u"], prep["i"])),
                        shape=prep["shape"], dtype=np.float32)
    model = AlternatingLeastSquares(
        factors=24, regularization=5.0, iterations=40,
        alpha=1.0, random_state=SEED, dtype=np.float32
    )
    model.fit(matrix)
    return model


def validate_steam(prep):
    observed = data_frame("steam_val_observed", ["user_id", "unified_game_id"])
    heldout = data_frame("steam_val_heldout", ["user_id", "unified_game_id"])
    validate_interactions(observed, "Steam Val Observed")
    validate_interactions(heldout, "Steam Val Heldout")
    val_matrix, val_users = build_validation_matrix(
        observed, prep["game_ids"], prep["game_to_idx"]
    )
    rows = []
    for mode in MODES:
        for beta in BETAS:
            print(f"START Steam mode={mode}, beta={beta}", flush=True)
            start = perf_counter()
            model = train_model(prep, mode, beta)
            scores = evaluate_all_users(
                model, val_matrix, val_users, observed, heldout,
                prep["game_ids"], k=10, batch_size=256
            )
            averages = scores[list(METRICS)].mean()
            row = {"mode": mode, "beta": beta, "users": len(scores),
                   **{key: float(averages[key]) for key in METRICS},
                   "seconds": round(perf_counter() - start, 2)}
            rows.append(row)
            print("RESULT", row, flush=True)
            pd.DataFrame(rows).to_csv(
                RESULTS / "als_median_gridsearch.csv",
                index=False, encoding="utf-8-sig"
            )
            del model, scores
    grid = pd.DataFrame(rows).sort_values("ndcg_at_10", ascending=False)
    print("=== STEAM GRID SORTED BY NDCG@10 ===", flush=True)
    print(grid.to_string(index=False), flush=True)
    return grid.iloc[0]


def test_ps(prep, best):
    mode, beta = str(best["mode"]), float(best["beta"])
    observed = data_frame("ps_test_observed", ["user_id", "unified_game_id"])
    heldout = data_frame("ps_test_heldout", ["user_id", "unified_game_id"])
    validate_interactions(observed, "PS Observed")
    validate_interactions(heldout, "PS Heldout")
    observed_by_user = observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    heldout_by_user = heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError("PS observed and heldout user sets differ")
    if any(not owned.isdisjoint(heldout_by_user[uid])
           for uid, owned in observed_by_user.items()):
        raise ValueError("PS observed/heldout overlap")
    ps_matrix, ps_users = build_ps_matrix(
        observed, prep["game_ids"], prep["game_to_idx"]
    )
    print(f"START PS external test: Steam-selected mode={mode}, beta={beta}", flush=True)
    start = perf_counter()
    model = train_model(prep, mode, beta)
    scores = evaluate_als(
        model, ps_matrix, ps_users, observed_by_user, heldout_by_user,
        prep["game_ids"], f"Median ALS {mode}/{beta}"
    )
    averages = scores[list(METRICS)].mean()
    row = {"mode": mode, "beta": beta, "users": len(scores),
           "steam_val_ndcg_at_10": float(best["ndcg_at_10"]),
           **{key: float(averages[key]) for key in METRICS},
           "seconds": round(perf_counter() - start, 2)}
    pd.DataFrame([row]).to_csv(
        RESULTS / "als_median_ps_test.csv", index=False, encoding="utf-8-sig"
    )
    print("=== PS TEST ===", row, flush=True)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = prepare()
    best = validate_steam(prep)
    test_ps(prep, best)
