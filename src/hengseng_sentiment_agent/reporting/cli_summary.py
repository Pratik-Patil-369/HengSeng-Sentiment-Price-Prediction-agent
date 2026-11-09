"""CLI formatted summary reporter using Rich tables."""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hengseng_sentiment_agent.agent.decision import TickerAnalysisResult
from hengseng_sentiment_agent.models.validation import ValidationScorecard

console = Console()


class CLISummaryReporter:
    """Prints beautiful formatted terminal reports."""

    @staticmethod
    def print_leaderboard(
        results: List[TickerAnalysisResult],
        sort_by: str = "social_mentions_count",
    ) -> None:
        """Print ranked table of assets by social mentions, sentiment, or direction."""
        sorted_results = sorted(
            results,
            key=lambda x: getattr(x, sort_by, 0),
            reverse=True,
        )

        table = Table(
            title="🎯 Multi-Source Sentiment Price Prediction Leaderboard",
            header_style="bold cyan",
            border_style="bright_black",
        )

        table.add_column("Ticker", style="bold white")
        table.add_column("Asset Name", style="white")
        table.add_column("Exch", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Pred (n+1)", justify="right", style="cyan")
        table.add_column("Pred Ret", justify="right")
        table.add_column("Sentiment", justify="center")
        table.add_column("Mentions", justify="right", style="bold")
        table.add_column("Direction", justify="center")
        table.add_column("Position", justify="center")
        table.add_column("Support / Resist", justify="center")
        table.add_column("SL / TP", justify="center")

        for r in sorted_results:
            is_bull = r.direction == "BULLISH"
            is_bear = r.direction == "BEARISH"

            dir_styled = (
                f"[bold green]▲ BULLISH[/bold green]"
                if is_bull
                else (f"[bold red]▼ BEARISH[/bold red]" if is_bear else "[yellow]● NEUTRAL[/yellow]")
            )
            pos_styled = (
                f"[green]{r.position} ({r.position_size_pct}%)[/green]"
                if r.position == "LONG"
                else (f"[red]{r.position} ({r.position_size_pct}%)[/red]" if r.position == "SHORT" else f"[dim]{r.position}[/dim]")
            )
            ret_styled = f"[green]+{r.predicted_return_pct:.2f}%[/green]" if r.predicted_return_pct >= 0 else f"[red]{r.predicted_return_pct:.2f}%[/red]"
            sent_styled = f"[green]+{r.sentiment_score:.3f}[/green]" if r.sentiment_score >= 0 else f"[red]{r.sentiment_score:.3f}[/red]"

            table.add_row(
                r.ticker,
                r.asset_name[:16],
                r.exchange,
                f"{r.current_price:,.2f}",
                f"{r.predicted_price_n1:,.2f}",
                ret_styled,
                sent_styled,
                str(r.social_mentions_count),
                dir_styled,
                pos_styled,
                f"[green]{r.support_resistance.nearest_support:,.1f}[/green] / [red]{r.support_resistance.nearest_resistance:,.1f}[/red]",
                f"[red]{r.stop_loss:,.1f}[/red] / [green]{r.take_profit:,.1f}[/green]",
            )

        console.print(table)

    @staticmethod
    def print_validation_scorecards(
        baseline_card: ValidationScorecard,
        sentiment_card: ValidationScorecard,
    ) -> None:
        """Print comparative out-of-sample model evaluation metrics."""
        table = Table(
            title="📊 Rigorous Out-of-Sample Walk-Forward Validation Scorecard",
            header_style="bold magenta",
            border_style="bright_black",
        )

        table.add_column("Model Metric", style="bold white")
        table.add_column(baseline_card.model_name, justify="center")
        table.add_column(sentiment_card.model_name, justify="center", style="bold cyan")
        table.add_column("Delta / Alpha Gain", justify="center", style="bold green")

        hit_diff = sentiment_card.directional_accuracy_pct - baseline_card.directional_accuracy_pct
        mae_diff = baseline_card.mae - sentiment_card.mae
        rmse_diff = baseline_card.rmse - sentiment_card.rmse
        ic_diff = sentiment_card.information_coefficient - baseline_card.information_coefficient
        pnl_diff = sentiment_card.cumulative_pnl_pct - baseline_card.cumulative_pnl_pct
        sharpe_diff = sentiment_card.sharpe_ratio - baseline_card.sharpe_ratio

        table.add_row(
            "Directional Accuracy (Hit Rate)",
            f"{baseline_card.directional_accuracy_pct:.1f}%",
            f"{sentiment_card.directional_accuracy_pct:.1f}%",
            f"+{hit_diff:.1f}%" if hit_diff >= 0 else f"{hit_diff:.1f}%",
        )
        table.add_row(
            "Mean Absolute Error (MAE)",
            f"{baseline_card.mae:.4f}",
            f"{sentiment_card.mae:.4f}",
            f"{mae_diff:+.4f} (improvement)",
        )
        table.add_row(
            "Root Mean Squared Error (RMSE)",
            f"{baseline_card.rmse:.4f}",
            f"{sentiment_card.rmse:.4f}",
            f"{rmse_diff:+.4f} (improvement)",
        )
        table.add_row(
            "Information Coefficient (Rank IC)",
            f"{baseline_card.information_coefficient:+.3f}",
            f"{sentiment_card.information_coefficient:+.3f}",
            f"{ic_diff:+.3f}",
        )
        table.add_row(
            "Cumulative Strategy Return",
            f"{baseline_card.cumulative_pnl_pct:+.2f}%",
            f"{sentiment_card.cumulative_pnl_pct:+.2f}%",
            f"+{pnl_diff:.2f}%",
        )
        table.add_row(
            "Annualized Sharpe Ratio",
            f"{baseline_card.sharpe_ratio:.2f}",
            f"{sentiment_card.sharpe_ratio:.2f}",
            f"+{sharpe_diff:.2f}",
        )

        console.print(table)
