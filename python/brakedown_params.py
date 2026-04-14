#!/usr/bin/env python3
"""
Compute Brakedown paper parameters for one or more (alpha, beta, r, n) choices.

This follows the formulas used in Section 5 / Algorithm 1 of:
https://eprint.iacr.org/2021/1043.pdf

Examples:
    python brakedown_params.py --alpha 0.238 --beta 0.1205 --r 1.72 --n 1073741824

    python brakedown_params.py \
        --alpha 0.20 0.211 0.238 \
        --beta 0.082 0.097 0.1205 \
        --r 1.64 1.616 1.72 \
        --n 1073741824 \
        --distance-gt 0.1 \
        --only-valid

    python brakedown_params.py \
        --alpha 0.30 0.305 0.31 0.315 0.32 \
        --beta 0.20 0.205 0.21 0.215 0.22 \
        --n 1024 \
        --near-min-r \
        --distance-gt 0.1 \
        --only-valid \
        --top-k 10
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class BrakedownAnalysis:
    alpha: float
    beta: float
    r: float
    n: int
    log2_q: int
    rate: float
    distance: float
    min_valid_r: float
    mu: float
    nu: float
    c_n: int
    d_n: int
    asymptotic_c: float
    asymptotic_d: float
    asymptotic_runtime_per_n: float
    satisfies_basic_constraints: bool


def binary_entropy(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"binary entropy is only defined on (0, 1), got {p}")
    one_minus_p = 1.0 - p
    return -p * math.log2(p) - one_minus_p * math.log2(one_minus_p)


def ceil(value: float) -> int:
    return math.ceil(value)


def min_valid_r(alpha: float, beta: float) -> float:
    return (1.0 + 2.0 * beta) / (1.0 - alpha)


def mu(alpha: float, r: float) -> float:
    return r - 1.0 - r * alpha


def nu(alpha: float, beta: float) -> float:
    return beta + alpha * beta + 0.03


def min_valid_r_from_mu_nu(alpha: float, beta: float) -> float:
    return (1.03 + beta * (1.0 + alpha)) / (1.0 - alpha)


def satisfies_basic_constraints(alpha: float, beta: float, r: float) -> bool:
    mu_value = mu(alpha, r)
    nu_value = nu(alpha, beta)
    return (
        0.0 < alpha < 1.0
        and 0.0 < beta < alpha / 1.28
        and r > min_valid_r(alpha, beta)
        and mu_value > 0.0
        and 0.0 < nu_value < mu_value
    )


def asymptotic_c(alpha: float, beta: float) -> float:
    return (
        binary_entropy(beta) + alpha * binary_entropy(1.28 * beta / alpha)
    ) / (beta * math.log2(alpha / (1.28 * beta)))


def asymptotic_d(alpha: float, beta: float, r: float) -> float:
    mu_value = mu(alpha, r)
    nu_value = nu(alpha, beta)
    return (
        r * alpha * binary_entropy(beta / r) + mu_value * binary_entropy(nu_value / mu_value)
    ) / (alpha * beta * math.log2(mu_value / nu_value))


def c_n(alpha: float, beta: float, n: int) -> int:
    n_f = float(n)
    return min(
        max(ceil(1.28 * beta * n_f), ceil(beta * n_f) + 4),
        ceil(
            (
                (110.0 / n_f)
                + binary_entropy(beta)
                + alpha * binary_entropy(1.28 * beta / alpha)
            )
            / (beta * math.log2(alpha / (1.28 * beta)))
        ),
    )


def d_n(alpha: float, beta: float, r: float, log2_q: int, n: int) -> int:
    mu_value = mu(alpha, r)
    nu_value = nu(alpha, beta)
    n_f = float(n)
    log2_q_f = float(log2_q)
    return min(
        ceil(2.0 * beta * n_f) + ceil(((r * n_f) - n_f + 110.0) / log2_q_f),
        ceil(
            (
                r * alpha * binary_entropy(beta / r)
                + mu_value * binary_entropy(nu_value / mu_value)
                + 110.0 / n_f
            )
            / (alpha * beta * math.log2(mu_value / nu_value))
        ),
    )


def analyze(alpha: float, beta: float, r: float, n: int, log2_q: int) -> BrakedownAnalysis:
    mu_value = mu(alpha, r)
    nu_value = nu(alpha, beta)
    return BrakedownAnalysis(
        alpha=alpha,
        beta=beta,
        r=r,
        n=n,
        log2_q=log2_q,
        rate=1.0 / r,
        distance=beta / r,
        min_valid_r=min_valid_r(alpha, beta),
        mu=mu_value,
        nu=nu_value,
        c_n=c_n(alpha, beta, n),
        d_n=d_n(alpha, beta, r, log2_q, n),
        asymptotic_c=asymptotic_c(alpha, beta),
        asymptotic_d=asymptotic_d(alpha, beta, r),
        asymptotic_runtime_per_n=(asymptotic_c(alpha, beta) + r * asymptotic_d(alpha, beta, r))
        / (1.0 - alpha),
        satisfies_basic_constraints=satisfies_basic_constraints(alpha, beta, r),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, nargs="+", required=True)
    parser.add_argument("--beta", type=float, nargs="+", required=True)
    parser.add_argument("--r", type=float, nargs="+")
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--log2-q", type=int, default=127)
    parser.add_argument("--distance-gt", type=float, default=None)
    parser.add_argument("--only-valid", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--near-min-r",
        action="store_true",
        help="derive r candidates near the minimum valid threshold instead of requiring --r",
    )
    parser.add_argument(
        "--r-margin",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        help="offsets added to the computed minimum feasible r when --near-min-r is used",
    )
    parser.add_argument("--csv", action="store_true", help="print CSV instead of aligned text")
    args = parser.parse_args()
    if not args.near_min_r and not args.r:
        parser.error("either provide --r or enable --near-min-r")
    return args


def sort_results(results: Iterable[BrakedownAnalysis]) -> list[BrakedownAnalysis]:
    return sorted(
        results,
        key=lambda item: (-item.distance, item.asymptotic_runtime_per_n, -item.rate),
    )


def filter_results(
    results: Iterable[BrakedownAnalysis],
    *,
    only_valid: bool,
    distance_gt: float | None,
    top_k: int | None,
) -> list[BrakedownAnalysis]:
    filtered = list(results)
    if only_valid:
        filtered = [item for item in filtered if item.satisfies_basic_constraints]
    if distance_gt is not None:
        filtered = [item for item in filtered if item.distance > distance_gt]
    filtered = sort_results(filtered)
    if top_k is not None:
        filtered = filtered[:top_k]
    return filtered


def generate_r_candidates(alpha: float, beta: float, margins: list[float]) -> list[float]:
    r_floor = max(
        min_valid_r(alpha, beta),
        min_valid_r_from_mu_nu(alpha, beta),
    )
    return [round(r_floor + margin, 12) for margin in margins]


def print_table(results: list[BrakedownAnalysis]) -> None:
    headers = [
        "alpha",
        "beta",
        "r",
        "n",
        "rate",
        "distance",
        "c_n",
        "d_n",
        "mu",
        "nu",
        "runtime/n",
        "valid",
    ]
    rows = [
        [
            f"{item.alpha:.4f}",
            f"{item.beta:.4f}",
            f"{item.r:.4f}",
            str(item.n),
            f"{item.rate:.4f}",
            f"{item.distance:.4f}",
            str(item.c_n),
            str(item.d_n),
            f"{item.mu:.4f}",
            f"{item.nu:.4f}",
            f"{item.asymptotic_runtime_per_n:.2f}",
            str(item.satisfies_basic_constraints),
        ]
        for item in results
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    print(" ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print(" ".join("-" * width for width in widths))
    for row in rows:
        print(" ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def print_csv(results: list[BrakedownAnalysis]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "alpha",
            "beta",
            "r",
            "n",
            "log2_q",
            "rate",
            "distance",
            "min_valid_r",
            "mu",
            "nu",
            "c_n",
            "d_n",
            "asymptotic_c",
            "asymptotic_d",
            "asymptotic_runtime_per_n",
            "satisfies_basic_constraints",
        ],
    )
    writer.writeheader()
    for item in results:
        writer.writerow(asdict(item))


def main() -> int:
    args = parse_args()

    if args.near_min_r:
        results = []
        for alpha, beta, n in product(args.alpha, args.beta, args.n):
            for r in generate_r_candidates(alpha, beta, args.r_margin):
                try:
                    results.append(analyze(alpha, beta, r, n, args.log2_q))
                except ValueError:
                    continue
    else:
        results = []
        for alpha, beta, r, n in product(args.alpha, args.beta, args.r, args.n):
            try:
                results.append(analyze(alpha, beta, r, n, args.log2_q))
            except ValueError:
                continue

    results = filter_results(
        results,
        only_valid=args.only_valid,
        distance_gt=args.distance_gt,
        top_k=args.top_k,
    )

    if args.csv:
        print_csv(results)
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
