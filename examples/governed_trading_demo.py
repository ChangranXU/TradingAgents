"""Comprehensive demonstration of TradingAgents with ArbiterOS Governance.

This example demonstrates:
1. ArbiterOS governance layer with policy checkers and routers
2. LangGraph checkpointing for workflow state persistence
3. Execution history tracking and visualization
4. Policy violation handling

Usage:
    # Run from the TradingAgents directory
    cd examples/TradingAgents
    python -m examples.governed_trading_demo

    # Or with custom parameters
    python -m examples.governed_trading_demo --ticker AAPL --date 2024-06-15

Requirements:
    - OPENAI_API_KEY environment variable
    - ALPHA_VANTAGE_API_KEY environment variable (for fundamental data)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.governed_agents import get_arbiter_os, reset_arbiter_os
from tradingagents.policies import (
    AnalystCompletionChecker,
    DebateRoundsChecker,
    RiskAnalysisChecker,
    ConfidenceRouter,
    RiskOverrideRouter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


def print_header(title: str) -> None:
    """Print a styled section header."""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold white]{title}[/bold white]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def setup_comprehensive_policies() -> None:
    """Configure comprehensive ArbiterOS policies for trading governance.

    This demonstrates various policy types:
    - AnalystCompletionChecker: Ensures all analysts complete before debate
    - DebateRoundsChecker: Enforces minimum debate rounds
    - RiskAnalysisChecker: Verifies risk team consensus
    """
    arbiter_os = get_arbiter_os()

    console.print("[bold]Configuring ArbiterOS Policies[/bold]\n")

    # Policy Checker 1: Analyst Completion
    analyst_checker = AnalystCompletionChecker(
        name="analyst_completion",
        required_analysts={"market", "social", "news", "fundamentals"},
    )
    arbiter_os.add_policy_checker(analyst_checker)
    console.print("[green]✓[/green] Added AnalystCompletionChecker")

    # Policy Checker 2: Minimum Debate Rounds
    debate_checker = DebateRoundsChecker(name="minimum_debate", min_rounds=1)
    arbiter_os.add_policy_checker(debate_checker)
    console.print("[green]✓[/green] Added DebateRoundsChecker (min_rounds=1)")

    # Policy Checker 3: Risk Analysis Verification
    risk_checker = RiskAnalysisChecker(name="risk_verification", min_risk_assessments=3)
    arbiter_os.add_policy_checker(risk_checker)
    console.print("[green]✓[/green] Added RiskAnalysisChecker (min_assessments=3)")

    console.print(f"\n[bold]Total policies configured:[/bold] {len(arbiter_os.policy_checkers)} checkers, {len(arbiter_os.policy_routers)} routers")


def create_config() -> dict:
    """Create the TradingAgents configuration.

    Returns:
        Configuration dictionary with LLM and data vendor settings.
    """
    config = DEFAULT_CONFIG.copy()
    config["deep_think_llm"] = "gpt-4o-mini"
    config["quick_think_llm"] = "gpt-4o-mini"
    config["max_debate_rounds"] = 1

    # Configure data vendors
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "alpha_vantage",
        "news_data": "alpha_vantage",
    }

    return config


def display_checkpoint_info(graph: TradingAgentsGraph, thread_id: str) -> None:
    """Display checkpoint information for a thread."""
    print_header("Checkpoint Information")

    checkpoints = graph.list_checkpoints(thread_id)

    if not checkpoints:
        console.print("[yellow]No checkpoints found for this thread.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Checkpoint ID", style="green")
    table.add_column("Step", style="yellow", justify="center")
    table.add_column("Next Nodes", style="white")

    for i, cp in enumerate(checkpoints[:10]):  # Show last 10
        cp_id = str(cp.get("checkpoint_id", "N/A"))
        if len(cp_id) > 20:
            cp_id = cp_id[:20] + "..."
        table.add_row(
            str(i + 1),
            cp_id,
            str(cp.get("step", "N/A")),
            ", ".join(cp.get("next_nodes", [])) or "None",
        )

    console.print(table)
    console.print(f"\n[dim]Total checkpoints: {len(checkpoints)}[/dim]")


def display_execution_history(graph: TradingAgentsGraph) -> None:
    """Display the ArbiterOS execution history."""
    print_header("ArbiterOS Execution History")

    history = graph.get_execution_history()

    if not history.entries:
        console.print("[yellow]No execution history recorded.[/yellow]")
        return

    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Superstep", style="cyan", justify="center")
    table.add_column("Instructions", style="green")
    table.add_column("Count", style="yellow", justify="center")

    for i, superstep in enumerate(history.entries):
        instructions = [item.instruction.name for item in superstep]
        instruction_summary = ", ".join(set(instructions))
        table.add_row(str(i + 1), instruction_summary, str(len(superstep)))

    console.print(table)

    # Print detailed history
    console.print("\n[bold]Detailed History:[/bold]")
    graph.print_execution_history()


def display_trading_decision(final_state: dict, decision: str) -> None:
    """Display the final trading decision."""
    print_header("Trading Decision")

    # Create decision panel
    decision_content = f"""
**Ticker:** {final_state.get('company_of_interest', 'N/A')}
**Date:** {final_state.get('trade_date', 'N/A')}

## Decision: {decision}

### Investment Plan Summary
{final_state.get('investment_plan', 'No plan available')[:500]}...

### Final Trade Decision
{final_state.get('final_trade_decision', 'No decision available')[:500]}...
"""

    console.print(Panel(
        Markdown(decision_content),
        title="Trading Decision",
        border_style="green",
    ))


def run_demo(
    ticker: str = "NVDA",
    trade_date: str = "2024-05-10",
    thread_id: Optional[str] = None,
    resume: bool = False,
    checkpoint_path: str = "./checkpoints/demo.db",
) -> None:
    """Run the complete governance demonstration.

    Args:
        ticker: Stock ticker symbol to analyze.
        trade_date: Date for analysis (YYYY-MM-DD).
        thread_id: Optional thread ID for checkpoint tracking.
        resume: If True, resume from existing checkpoint.
        checkpoint_path: Path to SQLite checkpoint database.
    """
    print_header("TradingAgents with ArbiterOS Governance Demo")

    console.print(f"[bold]Configuration:[/bold]")
    console.print(f"  Ticker: {ticker}")
    console.print(f"  Date: {trade_date}")
    console.print(f"  Checkpoint Path: {checkpoint_path}")
    console.print(f"  Resume: {resume}")

    # Reset ArbiterOS for fresh start (unless resuming)
    if not resume:
        reset_arbiter_os()
        console.print("\n[dim]ArbiterOS instance reset for fresh execution[/dim]")

    # Setup policies
    print_header("Policy Configuration")
    setup_comprehensive_policies()

    # Create configuration
    config = create_config()

    # Initialize TradingAgentsGraph
    print_header("Initializing TradingAgentsGraph")
    console.print("[dim]Creating graph with checkpointing enabled...[/dim]")

    graph = TradingAgentsGraph(
        debug=True,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    console.print("[green]✓[/green] Graph initialized successfully")
    console.print(f"[dim]Checkpointer: SqliteSaver @ {checkpoint_path}[/dim]")

    # Run the analysis
    print_header("Running Trading Analysis")
    console.print(f"[bold]Analyzing {ticker} for {trade_date}...[/bold]\n")

    try:
        final_state, decision = graph.propagate(
            ticker,
            trade_date,
            thread_id=thread_id,
            resume=resume,
        )

        # Store thread ID for later reference
        actual_thread_id = graph.current_thread_id
        console.print(f"\n[green]✓[/green] Analysis completed")
        console.print(f"[dim]Thread ID: {actual_thread_id}[/dim]")

        # Display results
        display_trading_decision(final_state, decision)
        display_checkpoint_info(graph, actual_thread_id)
        display_execution_history(graph)

        # Summary panel
        print_header("Demo Summary")
        summary = f"""
## ArbiterOS Governance Demo Complete

**Stock Analyzed:** {ticker}
**Analysis Date:** {trade_date}
**Decision:** {decision}

### Features Demonstrated:
1. ✓ Policy Checkers (AnalystCompletion, DebateRounds, RiskAnalysis)
2. ✓ LangGraph Checkpointing with SqliteSaver
3. ✓ Execution History Tracking
4. ✓ Thread-based State Management

### Next Steps:
- Resume this analysis: `--resume --thread-id {actual_thread_id}`
- List checkpoints: `--list-checkpoints {actual_thread_id}`
- Try a different ticker: `--ticker AAPL --date 2024-06-01`
"""
        console.print(Panel(Markdown(summary), border_style="cyan"))

    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        logger.exception("Analysis failed")
        raise


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="TradingAgents ArbiterOS Governance Demo"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol (default: NVDA)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-05-10",
        help="Trade date YYYY-MM-DD (default: 2024-05-10)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID for checkpoint tracking",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/demo.db",
        help="Path to checkpoint database",
    )
    parser.add_argument(
        "--list-checkpoints",
        type=str,
        metavar="THREAD_ID",
        help="List checkpoints for a thread and exit",
    )

    args = parser.parse_args()

    # Handle list checkpoints
    if args.list_checkpoints:
        config = create_config()
        graph = TradingAgentsGraph(
            debug=False,
            config=config,
            checkpoint_path=args.checkpoint_path,
        )
        display_checkpoint_info(graph, args.list_checkpoints)
        return

    # Run the demo
    run_demo(
        ticker=args.ticker,
        trade_date=args.date,
        thread_id=args.thread_id,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
