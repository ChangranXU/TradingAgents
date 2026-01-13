"""TradingAgents with ArbiterOS Governance.

This module demonstrates the TradingAgents framework with ArbiterOS governance
layer and LangGraph checkpointing for workflow state persistence.

Usage:
    # Basic usage (creates new checkpoint)
    python main.py

    # Resume from existing checkpoint
    python main.py --resume --thread-id <uuid>

    # List checkpoints for a thread
    python main.py --list-checkpoints --thread-id <uuid>
"""

import argparse
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.governed_agents import get_arbiter_os
from tradingagents.policies import (
    AnalystCompletionChecker,
    DebateRoundsChecker,
    RiskAnalysisChecker,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_policies() -> None:
    """Configure ArbiterOS policies for trading governance.

    This function sets up the policy checkers and routers that enforce
    trading workflow constraints.
    """
    arbiter_os = get_arbiter_os()

    # Add policy checkers
    arbiter_os.add_policy_checker(
        AnalystCompletionChecker(
            name="analyst_completion",
            required_analysts={"market", "social", "news", "fundamentals"},
        )
    )

    arbiter_os.add_policy_checker(
        DebateRoundsChecker(name="minimum_debate", min_rounds=1)
    )

    arbiter_os.add_policy_checker(
        RiskAnalysisChecker(name="risk_verification", min_risk_assessments=3)
    )

    logger.info("ArbiterOS policies configured")


def create_config() -> dict:
    """Create the TradingAgents configuration.

    Returns:
        Configuration dictionary with LLM and data vendor settings.
    """
    config = DEFAULT_CONFIG.copy()
    config["deep_think_llm"] = "gpt-4o-mini"
    config["quick_think_llm"] = "gpt-4o-mini"
    config["max_debate_rounds"] = 1

    # Configure data vendors (default uses yfinance and alpha_vantage)
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "alpha_vantage",
        "news_data": "alpha_vantage",
    }

    return config


def run_trading_analysis(
    ticker: str,
    trade_date: str,
    thread_id: Optional[str] = None,
    resume: bool = False,
    debug: bool = True,
    checkpoint_path: str = "./checkpoints/trading.db",
) -> None:
    """Run the trading analysis workflow.

    Args:
        ticker: Stock ticker symbol to analyze.
        trade_date: Date for the trading analysis (YYYY-MM-DD).
        thread_id: Optional thread ID for checkpoint tracking.
        resume: If True, resume from existing checkpoint.
        debug: Enable debug mode with verbose output.
        checkpoint_path: Path to SQLite checkpoint database.
    """
    # Setup policies
    setup_policies()

    # Create configuration
    config = create_config()

    # Initialize TradingAgentsGraph with checkpointing
    logger.info(f"Initializing TradingAgents with checkpoint: {checkpoint_path}")
    ta = TradingAgentsGraph(
        debug=debug,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    # Run the workflow
    logger.info(f"Analyzing {ticker} for {trade_date}")
    if thread_id:
        logger.info(f"Thread ID: {thread_id}")
    if resume:
        logger.info("Resuming from checkpoint...")

    final_state, decision = ta.propagate(
        ticker, trade_date, thread_id=thread_id, resume=resume
    )

    # Print results
    print("\n" + "=" * 60)
    print("TRADING DECISION")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Date: {trade_date}")
    print(f"Decision: {decision}")
    print(f"Thread ID: {ta.current_thread_id}")
    print("=" * 60)

    # Print execution history
    print("\n" + "=" * 60)
    print("ARBITER OS EXECUTION HISTORY")
    print("=" * 60)
    ta.print_execution_history()


def list_checkpoints(thread_id: str, checkpoint_path: str) -> None:
    """List all checkpoints for a given thread ID.

    Args:
        thread_id: Thread ID to list checkpoints for.
        checkpoint_path: Path to SQLite checkpoint database.
    """
    config = create_config()
    ta = TradingAgentsGraph(
        debug=False,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    checkpoints = ta.list_checkpoints(thread_id)

    print(f"\nCheckpoints for thread: {thread_id}")
    print("=" * 60)

    if not checkpoints:
        print("No checkpoints found.")
        return

    for i, cp in enumerate(checkpoints):
        print(f"\n[{i + 1}] Checkpoint ID: {cp.get('checkpoint_id', 'N/A')}")
        print(f"    Timestamp: {cp.get('timestamp', 'N/A')}")
        print(f"    Step: {cp.get('step', 'N/A')}")
        print(f"    Next Nodes: {cp.get('next_nodes', [])}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="TradingAgents with ArbiterOS Governance"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol to analyze (default: NVDA)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-05-10",
        help="Trade date in YYYY-MM-DD format (default: 2024-05-10)",
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
        "--list-checkpoints",
        action="store_true",
        help="List checkpoints for the given thread ID",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/trading.db",
        help="Path to checkpoint database (default: ./checkpoints/trading.db)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode (default: True)",
    )

    args = parser.parse_args()

    if args.list_checkpoints:
        if not args.thread_id:
            print("Error: --thread-id required with --list-checkpoints")
            sys.exit(1)
        list_checkpoints(args.thread_id, args.checkpoint_path)
    else:
        run_trading_analysis(
            ticker=args.ticker,
            trade_date=args.date,
            thread_id=args.thread_id,
            resume=args.resume,
            debug=args.debug,
            checkpoint_path=args.checkpoint_path,
        )


if __name__ == "__main__":
    main()
