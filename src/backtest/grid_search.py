"""Hyperparameter grid search for strategy optimization."""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.backtest.types import BacktestResult
from src.config.logging import get_logger
from src.strategy.params import StrategyParams

logger = get_logger(__name__)


def run_grid_search(
    engine_factory: Callable[[StrategyParams, str], BacktestEngine],
    train_start: str,
    train_end: str,
    param_grid: Dict[str, List[Any]],
    instrument: str,
    df_all: pd.DataFrame,
    df_daily_atr: Optional[pd.DataFrame] = None,
    df_intraday_fallback: Optional[pd.DataFrame] = None,
    selection_metric: str = "sharpe",
) -> Dict[str, Any]:
    """Run hyperparameter grid search on training period.

    Args:
        engine_factory: Function that takes StrategyParams and returns BacktestEngine
        train_start: Training start date (YYYY-MM-DD)
        train_end: Training end date (YYYY-MM-DD)
        param_grid: Dictionary mapping parameter names to lists of values
            Example: {"rr": [1.5, 2.0, 2.5], "atr_mult": [1.0, 1.2, 1.5]}
        instrument: Instrument identifier
        df_all: Full DataFrame with main timeframe data
        df_daily_atr: Optional DataFrame with daily candles for ATR calculation
        df_intraday_fallback: Optional intraday data to resample if daily unavailable
        selection_metric: Metric to use for selecting best params ("sharpe" or "total_R")

    Returns:
        Dictionary with:
        - best_params: Best StrategyParams instance
        - best_metrics: Metrics for best parameter set
        - all_results: List of all (params, metrics) tuples
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Create base params with defaults
    base_params = StrategyParams()

    all_results: List[tuple[StrategyParams, Dict[str, float]]] = []

    # Iterate over all combinations
    from itertools import product

    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    logger.info(f"Running grid search with {total_combinations} parameter combinations...")

    for i, combination in enumerate(product(*param_values), 1):
        # Create params dict with base values
        params_dict = {
            "or_start_time": base_params.or_start_time,
            "or_end_time": base_params.or_end_time,
            "rr": base_params.rr,
            "atr_period": base_params.atr_period,
            "atr_mult": base_params.atr_mult,
            "use_mid_stop": base_params.use_mid_stop,
            "tp_sl_conflict_policy": base_params.tp_sl_conflict_policy,
            "risk_per_trade": base_params.risk_per_trade,
            "max_daily_risk": base_params.max_daily_risk,
            "slippage_model": base_params.slippage_model,
            "fee_model": base_params.fee_model,
        }

        # Override with grid values
        for name, value in zip(param_names, combination):
            params_dict[name] = value

        # Create StrategyParams
        try:
            params = StrategyParams(**params_dict)
        except ValueError as e:
            logger.warning(f"Skipping invalid parameter combination {combination}: {e}")
            continue

        # Create engine and run backtest
        try:
            engine = engine_factory(params, instrument)
            result = engine.run(
                df_all=df_all,
                df_daily_atr=df_daily_atr,
                df_intraday_fallback=df_intraday_fallback,
                start_date=train_start,
                end_date=train_end,
            )

            # Get metrics
            if result.trades:
                trades_df = pd.DataFrame([trade.to_dict() for trade in result.trades])
                metrics = compute_metrics(trades_df)
            else:
                metrics = compute_metrics(pd.DataFrame())

            all_results.append((params, metrics))

            logger.debug(
                f"Combination {i}/{total_combinations}: {dict(zip(param_names, combination))} "
                f"-> Sharpe: {metrics.get('sharpe', 0.0):.3f}, Total R: {metrics.get('total_R', 0.0):.3f}"
            )

        except Exception as e:
            logger.warning(f"Error running combination {combination}: {e}")
            continue

    if not all_results:
        raise ValueError("No valid parameter combinations found")

    # Select best based on selection metric
    if selection_metric == "sharpe":
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i][1].get("sharpe", -999.0))
    elif selection_metric == "total_R":
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i][1].get("total_R", -999.0))
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    best_params, best_metrics = all_results[best_idx]

    logger.info(f"Best parameters (based on {selection_metric}):")
    for name in param_names:
        value = getattr(best_params, name)
        logger.info(f"  {name}: {value}")
    logger.info(f"Metrics: Sharpe={best_metrics.get('sharpe', 0.0):.3f}, Total R={best_metrics.get('total_R', 0.0):.3f}")

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_results": all_results,
    }

