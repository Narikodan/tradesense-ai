from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestTrade:
    signal_date: object
    direction: str
    entry: float
    stop_loss: float
    target1: float
    outcome: str
    fill_date: object | None = None
    exit_date: object | None = None
    r_multiple: float = 0.0


@dataclass(frozen=True)
class BacktestSummary:
    trades: int
    wins: int
    losses: int
    open: int
    no_fill: int
    win_rate: float
    average_r: float
    total_r: float


def simulate_trade(signal: dict, future: pd.DataFrame) -> BacktestTrade:
    """
    Deterministic educational trade simulation.

    Entry must fill before a stop/target can count. If a single candle touches
    both stop and target after fill, the conservative stop-first outcome is used.
    This is not brokerage-grade execution modeling.
    """
    direction = signal.get("recommendation")
    if direction not in {"BUY", "SELL", "SHORT_SELL"}:
        raise ValueError("signal recommendation must be BUY, SELL, or SHORT_SELL")
    if future is None or future.empty:
        return _trade(signal, direction, "open")

    entry = float(signal["entry"])
    stop = float(signal["stop_loss"])
    target = float(signal["target1"])
    risk = abs(entry - stop)
    filled = False
    fill_date = None

    for ts, row in future.iterrows():
        high = float(row["High"])
        low = float(row["Low"])

        if not filled:
            filled = (low <= entry <= high)
            if not filled:
                continue
            fill_date = ts

        if direction == "BUY":
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop:
            return _trade(signal, direction, "stop", fill_date, ts, -1.0)
        if hit_target:
            r_multiple = round(abs(target - entry) / risk, 2) if risk > 0 else 0.0
            return _trade(signal, direction, "target", fill_date, ts, r_multiple)

    if not filled:
        return _trade(signal, direction, "no_fill")
    return _trade(signal, direction, "open", fill_date, None, 0.0)


def summarize_trades(trades: list[BacktestTrade]) -> BacktestSummary:
    wins = sum(1 for trade in trades if trade.outcome == "target")
    losses = sum(1 for trade in trades if trade.outcome == "stop")
    open_ = sum(1 for trade in trades if trade.outcome == "open")
    no_fill = sum(1 for trade in trades if trade.outcome == "no_fill")
    closed = wins + losses
    total_r = round(sum(trade.r_multiple for trade in trades), 2)
    return BacktestSummary(
        trades=len(trades),
        wins=wins,
        losses=losses,
        open=open_,
        no_fill=no_fill,
        win_rate=round((wins / closed) * 100, 2) if closed else 0.0,
        average_r=round(total_r / len(trades), 2) if trades else 0.0,
        total_r=total_r,
    )


def walk_forward_validate(analyzer_cls, df: pd.DataFrame, *, window: int = 120, step: int = 5) -> list[BacktestTrade]:
    """
    Walk-forward validation scaffold using deterministic entry/exit simulation.

    It is intentionally conservative and educational: fills are candle-based,
    no slippage/fees are modeled, and same-candle stop/target ambiguity resolves
    to stop-first.
    """
    trades: list[BacktestTrade] = []
    if df is None or len(df) < window + step:
        return trades

    for end in range(window, len(df) - step, step):
        history = df.iloc[:end]
        future = df.iloc[end : end + step]
        result = analyzer_cls(history).analyze()
        if result.get("recommendation") not in {"BUY", "SELL", "SHORT_SELL"}:
            continue
        trade = simulate_trade({"signal_date": history.index[-1], **result}, future)
        trades.append(trade)
    return trades


def _trade(
    signal: dict,
    direction: str,
    outcome: str,
    fill_date=None,
    exit_date=None,
    r_multiple: float = 0.0,
) -> BacktestTrade:
    return BacktestTrade(
        signal_date=signal.get("signal_date"),
        direction=direction,
        entry=round(float(signal["entry"]), 2),
        stop_loss=round(float(signal["stop_loss"]), 2),
        target1=round(float(signal["target1"]), 2),
        outcome=outcome,
        fill_date=fill_date,
        exit_date=exit_date,
        r_multiple=r_multiple,
    )
