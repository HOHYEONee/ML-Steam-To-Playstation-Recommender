"""Sweep ownership confidence alpha for game-median ALS.

A: penalty below each game's median, including zero.
B: bonus for above-median time only.
C: penalty below median for positive playtime; zero stays neutral.

Run from repository root:
    python src/models/als/als_median_alpha_sweep.py --phase grid --alpha 10
    python src/models/als/als_median_alpha_sweep.py --phase ps
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import als_median_experiment as median

METRICS = ("recall_at_10", "ndcg_at_10", "hit_at_10")
DATA = median.DATA
RESULTS = median.RESULTS


def evaluate_grid(alpha):
    """Validate all nine median configurations and a same-alpha binary control."""
    median.ALPHA = float(alpha)
    prep = median.prepare()

    observed = median.data_frame("steam_val_observed", ["user_id", "unified_game_id"])
    heldout = median.data_frame("steam_val_heldout", ["user_id", "unified_game_id"])
    median.validate_interactions(observed, "Steam Val Observed")
    median.validate_interactions(heldout, "Steam Val Heldout")
    val_matrix, val_users = median.build_validation_matrix(
        observed, prep["game_ids"], prep["game_to_idx"]
    )
    combinations = [("Binary", 0.0)]
    combinations += [(m, b) for m in median.MODES for b in median.BETAS]

    rows = []
    for mode, beta in combinations:
        print(f"START alpha={alpha}, mode={mode}, beta={beta}", flush=True)
        t0 = perf_counter()
        # beta=0 and Mode B gives identical 1 + alpha for all owned interactions
        model = median.train_model(prep, "B" if mode == "Binary" else mode, beta)
        user_metrics = median.evaluate_all_users(
            model, val_matrix, val_users, observed, heldout,
            prep["game_ids"], k=10, batch_size=256
        )
        averages = user_metrics[list(METRICS)].mean()
        row = {
            "alpha": float(alpha), "mode": mode, "beta": beta,
            "users": len(user_metrics),
            **{m: float(averages[m]) for m in METRICS},
            "seconds": round(perf_counter() - t0, 2)
        }
        rows.append(row)
        print("RESULT", row, flush=True)
        # save after each combination, in case the workflow is interrupted
        pd.DataFrame(rows).to_csv(
            RESULTS / f"als_median_alpha_{int(alpha)}_grid.csv",
            index=False, encoding="utf-8-sig"
        )
        del model, user_metrics
    return pd.DataFrame(rows)


def evaluate_ps():
    """Select the median model on Steam validation, then test it and same-alpha Binary."""
    paths = sorted(RESULTS.glob("als_median_alpha_*_grid.csv"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four alpha-grid CSV files, found {len(paths)}: {paths}")
    grid = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    grid = grid.sort_values("ndcg_at_10", ascending=False)
    grid.to_csv(RESULTS / "als_median_alpha_full_grid.csv",
                index=False, encoding="utf-8-sig")
    print("STEAM SUMMARY BY NDCG", flush=True)
    print(grid.to_string(index=False), flush=True)
    # Choose among A/B/C using ONLY Steam validation. Evaluate binary at same alpha.
    eligible = grid.loc[grid["mode"] != "Binary"]
    chosen = eligible.iloc[0]
    alpha = float(chosen["alpha"])
    mode = str(chosen["mode"])
    beta = float(chosen["beta"])
    print(f"STEAM SELECTED alpha={alpha}, mode={mode}, beta={beta}", flush=True)

    median.ALPHA = alpha
    prep = median.prepare()
    observed = median.data_frame("ps_test_observed", ["user_id", "unified_game_id"])
    heldout = median.data_frame("ps_test_heldout", ["user_id", "unified_game_id"])
    median.validate_interactions(observed, "PS Observed")
    median.validate_interactions(heldout, "PS Heldout")
    observed_by_user = observed.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    heldout_by_user = heldout.groupby("user_id")["unified_game_id"].agg(set).to_dict()
    if set(observed_by_user) != set(heldout_by_user):
        raise ValueError("PS observed and heldout user sets differ")
    if any(not games.isdisjoint(heldout_by_user[uid])
           for uid, games in observed_by_user.items()):
        raise ValueError("PS observed and heldout overlap")
    ps_matrix, ps_users = median.build_ps_matrix(
        observed, prep["game_ids"], prep["game_to_idx"]
    )

    results = []
    for candidate_mode, candidate_beta in [(mode, beta), ("Binary", 0.0)]:
        print(f"START PS alpha={alpha} mode={candidate_mode} beta={candidate_beta}", flush=True)
        t0 = perf_counter()
        model = median.train_model(
            prep, "B" if candidate_mode == "Binary" else candidate_mode, candidate_beta
        )
        scores = median.evaluate_als(
            model, ps_matrix, ps_users, observed_by_user, heldout_by_user,
            prep["game_ids"], f"Median alpha={alpha} {candidate_mode}/{candidate_beta}"
        )
        averages = scores[list(METRICS)].mean()
        row = {
            "alpha": alpha,
            "mode": candidate_mode,
            "beta": candidate_beta,
            "users": len(scores),
            "selected_steam_ndcg_at_10": float(chosen["ndcg_at_10"]),
            **{m: float(averages[m]) for m in METRICS},
            "seconds": round(perf_counter() - t0, 2)
        }
        results.append(row)
        print("PS RESULT", row, flush=True)
        pd.DataFrame(results).to_csv(
            RESULTS / "als_median_alpha_ps_test.csv",
            index=False, encoding="utf-8-sig"
        )
        del model, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["grid", "ps"], required=True)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.phase == "grid":
        if args.alpha not in (2.0, 5.0, 10.0, 20.0):
            parser.error("grid --alpha must be one of: 2, 5, 10, 20")
        evaluate_grid(args.alpha)
    else:
        evaluate_ps()


if __name__ == "__main__":
    main()
