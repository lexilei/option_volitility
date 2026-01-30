"""Dashboard components."""

from .charts import (
    create_candlestick_chart,
    create_line_chart,
    create_equity_curve,
    create_drawdown_chart,
    create_returns_histogram,
    create_correlation_heatmap,
    create_bar_chart,
    create_monthly_heatmap,
    create_feature_importance_chart,
    create_subplots_chart,
)

from .tables import (
    format_percentage,
    format_currency,
    format_number,
    create_metrics_table,
    display_performance_metrics,
    create_model_comparison_table,
    create_trades_table,
    create_summary_stats_table,
    create_correlation_table,
    display_dataframe_with_pagination,
)

__all__ = [
    # Charts
    "create_candlestick_chart",
    "create_line_chart",
    "create_equity_curve",
    "create_drawdown_chart",
    "create_returns_histogram",
    "create_correlation_heatmap",
    "create_bar_chart",
    "create_monthly_heatmap",
    "create_feature_importance_chart",
    "create_subplots_chart",
    # Tables
    "format_percentage",
    "format_currency",
    "format_number",
    "create_metrics_table",
    "display_performance_metrics",
    "create_model_comparison_table",
    "create_trades_table",
    "create_summary_stats_table",
    "create_correlation_table",
    "display_dataframe_with_pagination",
]
