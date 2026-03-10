"""Command line interface."""
from __future__ import annotations

import argparse

import pandas as pd
import yaml

from .backtest import walk_forward_backtest
from .data import align_calendar, fetch_prices, save_prices_parquet
from .metrics import max_drawdown, sharpe, sortino
from .pairs import select_pairs
from .portfolio import apply_limits, neutralize_beta, neutralize_by_sector
from .signals import compute_zscore, generate_pair_positions
from .kalman import kalman_regression
from .portfolio import aggregate_pair_weights
from .universe import load_sector_map, load_universe
from .dashboard import plot_data_coverage, plot_pnl_and_positions, save_coverage_csv
from execution.rebalance import targets_to_orders
from execution.risk_manager import check_risk


def _cmd_download_data(config_path: str, output_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = load_universe(config_path)
    prices_cfg = cfg.get("prices", {})
    start = prices_cfg.get("start")
    end = prices_cfg.get("end")
    source = prices_cfg.get("source", "yfinance")
    if not start or not end:
        raise ValueError("prices.start and prices.end are required in config")

    df = fetch_prices(symbols, start=start, end=end, source=source)
    max_missing_pct = prices_cfg.get("max_missing_pct", 0.2)
    res = align_calendar(df, max_missing_pct=max_missing_pct)
    save_prices_parquet(res.prices, output_path)
    print("Saved:", output_path)
    print("Missing report (top 10):")
    print(res.missing_report.head(10))


def _as_float(value, default: float) -> float:
    """Parse numeric config values that may be strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return float(value)
    return float(value)


def _cmd_select_pairs(config_path: str, prices_path: str, output_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    universe_cfg_path = cfg.get("universe_config")
    if not universe_cfg_path:
        raise ValueError("universe_config is required in backtest config")
    with open(universe_cfg_path, "r", encoding="utf-8") as f:
        uni_cfg = yaml.safe_load(f)
    sector_cfg = uni_cfg.get("sector_map", {})
    sector_source = sector_cfg.get("source")
    sector_path = sector_cfg.get("path")
    if not sector_source or not sector_path:
        raise ValueError("sector_map.source and sector_map.path are required in universe config")

    prices = pd.read_parquet(prices_path)
    sector_map = load_sector_map(sector_source, sector_path)

    pairs = select_pairs(
        prices=prices,
        sector_map=sector_map,
        train_window=cfg.get("train_window", 252),
        corr_lookback=cfg.get("corr_lookback", 252),
        corr_threshold=cfg.get("corr_threshold"),
        min_half_life=cfg.get("half_life_min", 2),
        max_half_life=cfg.get("half_life_max", 20),
        pval_thresh=cfg.get("pval_thresh", 0.05),
        max_pairs=cfg.get("max_pairs", 50),
        rank_by=cfg.get("rank_by", "pvalue"),
    )
    rows = [
        {
            "y": p.y,
            "x": p.x,
            "sector": p.sector,
            "init_beta": p.init_beta,
            "half_life": p.half_life,
            "pvalue": p.pvalue,
            "crossings": p.crossings,
        }
        for p in pairs
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print(out.head(10).to_string(index=False))


def _cmd_backtest(config_path: str, prices_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    universe_cfg_path = cfg.get("universe_config")
    if not universe_cfg_path:
        raise ValueError("universe_config is required in backtest config")
    with open(universe_cfg_path, "r", encoding="utf-8") as f:
        uni_cfg = yaml.safe_load(f)
    sector_cfg = uni_cfg.get("sector_map", {})
    sector_map = load_sector_map(sector_cfg.get("source"), sector_cfg.get("path"))

    prices = pd.read_parquet(prices_path)
    spy_symbol = cfg.get("spy_symbol", "SPY")
    if spy_symbol not in prices.columns:
        raise ValueError(f"SPY symbol {spy_symbol} not in prices")
    spy = prices[spy_symbol]

    equity, pair_positions, weights = walk_forward_backtest(
        prices=prices,
        sector_map=sector_map,
        spy=spy,
        train_window=cfg.get("train_window", 252),
        test_step=cfg.get("step_window", 21),
        corr_lookback=cfg.get("corr_lookback", 252),
        corr_threshold=cfg.get("corr_threshold"),
        max_pairs=cfg.get("max_pairs", 50),
        pval_thresh=cfg.get("pval_thresh", 0.05),
        half_life_min=cfg.get("half_life_min", 2),
        half_life_max=cfg.get("half_life_max", 20),
        rank_by=cfg.get("rank_by", "pvalue"),
        kalman_R=_as_float(cfg.get("kalman", {}).get("R"), 1e-3),
        kalman_Q=_as_float(cfg.get("kalman", {}).get("Q"), 1e-4),
        entry_z=cfg.get("signals", {}).get("entry_z", 2.0),
        exit_z=cfg.get("signals", {}).get("exit_z", 0.5),
        stop_z=cfg.get("signals", {}).get("stop_z", 4.0),
        max_pair_w=cfg.get("portfolio", {}).get("max_pair_w", 0.02),
        w_max=cfg.get("portfolio", {}).get("w_max", 0.02),
        gross_max=cfg.get("portfolio", {}).get("gross_max", 1.0),
        slippage_bps=_as_float(cfg.get("costs", {}).get("slippage_bps"), 2.0),
        proportional=cfg.get("signals", {}).get("proportional", False),
        weight_smooth=cfg.get("portfolio", {}).get("weight_smooth", 0.0),
    )

    returns = equity.pct_change().dropna()
    print("CAGR:", (equity.iloc[-1] ** (252 / len(equity)) - 1.0))
    print("Sharpe:", sharpe(returns))
    print("Sortino:", sortino(returns))
    print("MaxDD:", max_drawdown(equity))

    equity.to_csv("data/processed/equity.csv")
    weights.to_parquet("data/processed/weights.parquet")
    pair_positions.to_parquet("data/processed/pair_positions.parquet")
    print("Saved: data/processed/equity.csv")


def _cmd_generate_weights(config_path: str, prices_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    universe_cfg_path = cfg.get("universe_config")
    if not universe_cfg_path:
        raise ValueError("universe_config is required in live config")
    with open(universe_cfg_path, "r", encoding="utf-8") as f:
        uni_cfg = yaml.safe_load(f)
    sector_cfg = uni_cfg.get("sector_map", {})
    sector_map = load_sector_map(sector_cfg.get("source"), sector_cfg.get("path"))

    prices = pd.read_parquet(prices_path)
    spy_symbol = cfg.get("spy_symbol", "SPY")
    spy = prices[spy_symbol]

    train_window = cfg.get("train_window", 252)
    if len(prices) < train_window:
        raise ValueError("Not enough data for train_window")
    train_prices = prices.tail(train_window)
    pairs = select_pairs(
        prices=train_prices,
        sector_map=sector_map,
        train_window=train_window,
        corr_lookback=cfg.get("corr_lookback", 252),
        corr_threshold=cfg.get("corr_threshold"),
        min_half_life=cfg.get("half_life_min", 2),
        max_half_life=cfg.get("half_life_max", 20),
        pval_thresh=cfg.get("pval_thresh", 0.05),
        max_pairs=cfg.get("max_pairs", 50),
        rank_by=cfg.get("rank_by", "pvalue"),
    )

    latest_date = prices.index[-1]
    raw_weights = pd.Series(dtype=float)
    betas_t = {}
    pair_positions = {}

    for pair in pairs:
        y = prices[pair.y]
        x = prices[pair.x]
        _, beta_s, spread = kalman_regression(
            y,
            x,
            R=_as_float(cfg.get("kalman", {}).get("R"), 1e-3),
            Q=_as_float(cfg.get("kalman", {}).get("Q"), 1e-4),
        )
        lookback = max(10, pair.half_life * 2)
        z = compute_zscore(spread, lookback=lookback)
        pos = generate_pair_positions(
            z,
            entry=cfg.get("signals", {}).get("entry_z", 2.0),
            exit=cfg.get("signals", {}).get("exit_z", 0.5),
            stop=cfg.get("signals", {}).get("stop_z", 4.0),
        )
        key = f"{pair.y}-{pair.x}"
        pair_positions[key] = float(pos.loc[latest_date])
        betas_t[key] = float(beta_s.loc[latest_date])

        raw = aggregate_pair_weights(
            pairs=[pair],
            pair_positions={key: pair_positions[key]},
            betas_t={key: betas_t[key]},
            max_pair_w=cfg.get("portfolio", {}).get("max_pair_w", 0.02),
        )
        raw_weights = raw_weights.add(raw, fill_value=0.0)

    # Estimate betas using trailing window
    returns = prices.pct_change().dropna()
    spy_ret = spy.pct_change().dropna()
    if len(returns) >= train_window:
        train_ret = returns.tail(train_window)
        spy_train = spy_ret.tail(train_window)
        cov = train_ret.apply(lambda x: x.cov(spy_train))
        betas = cov / spy_train.var(ddof=0)
    else:
        betas = pd.Series(0.0, index=raw_weights.index)

    w = neutralize_by_sector(raw_weights.fillna(0.0), sector_map)
    w = neutralize_beta(w, betas, target_beta=0.0)
    w = apply_limits(
        w,
        w_max=cfg.get("portfolio", {}).get("w_max", 0.02),
        gross_max=cfg.get("portfolio", {}).get("gross_max", 1.0),
    )
    out = pd.DataFrame({"symbol": w.index, "weight": w.values})
    out.to_csv("data/processed/target_weights.csv", index=False)
    print("Saved: data/processed/target_weights.csv")


def _cmd_trade_paper(config_path: str, prices_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dry_run = cfg.get("execution", {}).get("dry_run", True)
    max_daily_loss_pct = cfg.get("execution", {}).get("max_daily_loss_pct", 0.02)
    gross_max = cfg.get("portfolio", {}).get("gross_max", 1.0)
    net_max = cfg.get("portfolio", {}).get("net_max", 0.2)

    target_df = pd.read_csv("data/processed/target_weights.csv")
    target_weights = dict(zip(target_df["symbol"], target_df["weight"]))

    if not check_risk(target_weights, gross_max=gross_max, net_max=net_max, max_daily_loss_pct=max_daily_loss_pct):
        raise SystemExit("Risk check failed; aborting")

    prices = pd.read_parquet(prices_path)
    latest_prices = prices.iloc[-1]
    equity = float(cfg.get("execution", {}).get("equity", 100000.0))
    current_positions = {}
    orders = targets_to_orders(target_weights, latest_prices, current_positions, equity)

    if dry_run:
        print("Dry run orders:", orders)
        return
    raise SystemExit("live submit not implemented yet")


def _cmd_dashboard_data(prices_path: str, output_dir: str) -> None:
    prices = pd.read_parquet(prices_path)
    plot_data_coverage(prices, output_dir=output_dir)
    save_coverage_csv(prices, output_path=f"{output_dir}/coverage_summary.csv")
    print("Saved data coverage dashboard to:", output_dir)


def _cmd_dashboard_pnl(
    equity_path: str,
    weights_path: str,
    output_dir: str,
) -> None:
    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True).iloc[:, 0]
    weights = pd.read_parquet(weights_path)
    plot_pnl_and_positions(equity, weights, output_dir=output_dir)
    print("Saved PnL/positions dashboard to:", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(prog="statarb")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download-data")
    dl.add_argument("--config", default="configs/universe.yaml")
    dl.add_argument("--output", default="data/processed/prices.parquet")

    sp = sub.add_parser("select-pairs")
    sp.add_argument("--config", default="configs/backtest.yaml")
    sp.add_argument("--prices", default="data/processed/prices.parquet")
    sp.add_argument("--output", default="data/processed/pairs.csv")

    bt = sub.add_parser("backtest")
    bt.add_argument("--config", default="configs/backtest.yaml")
    bt.add_argument("--prices", default="data/processed/prices.parquet")

    gw = sub.add_parser("generate-weights")
    gw.add_argument("--config", default="configs/live.yaml")
    gw.add_argument("--prices", default="data/processed/prices.parquet")

    tp = sub.add_parser("trade-paper")
    tp.add_argument("--config", default="configs/live.yaml")
    tp.add_argument("--prices", default="data/processed/prices.parquet")

    dd = sub.add_parser("dashboard-data")
    dd.add_argument("--prices", default="data/processed/prices.parquet")
    dd.add_argument("--output", default="data/processed/dashboard/data")

    dp = sub.add_parser("dashboard-pnl")
    dp.add_argument("--equity", default="data/processed/equity.csv")
    dp.add_argument("--weights", default="data/processed/weights.parquet")
    dp.add_argument("--output", default="data/processed/dashboard/pnl")

    args = parser.parse_args()
    if args.cmd == "download-data":
        _cmd_download_data(args.config, args.output)
        return
    if args.cmd == "select-pairs":
        _cmd_select_pairs(args.config, args.prices, args.output)
        return
    if args.cmd == "backtest":
        _cmd_backtest(args.config, args.prices)
        return
    if args.cmd == "generate-weights":
        _cmd_generate_weights(args.config, args.prices)
        return
    if args.cmd == "trade-paper":
        _cmd_trade_paper(args.config, args.prices)
        return
    if args.cmd == "dashboard-data":
        _cmd_dashboard_data(args.prices, args.output)
        return
    if args.cmd == "dashboard-pnl":
        _cmd_dashboard_pnl(args.equity, args.weights, args.output)
        return
    raise SystemExit(f"Not implemented yet: {args.cmd}")
