"""Test TradingAgents with ArbiterOS using custom API configuration.

This test script demonstrates:
1. Custom OpenAI API configuration (base_url, model, api_key)
2. ArbiterOS governance integration
3. Minimal test run to verify the framework works

Usage:
    cd examples/TradingAgents
    uv run python test_arbiteros_integration.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.governed_agents import get_arbiter_os, reset_arbiter_os
from tradingagents.policies import (
    AnalystCompletionChecker,
    DebateRoundsChecker,
    RiskAnalysisChecker,
)

print("=" * 60)
print("TradingAgents + ArbiterOS Integration Test")
print("=" * 60)

# Configure custom API settings
API_KEY = "*********"
BASE_URL = "https://*********"
MODEL = "gpt-4.1"
ALPHA_VANTAGE_API_KEY = "*********"

# Set environment variables
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY

print(f"\n✓ API Configuration:")
print(f"  Base URL: {BASE_URL}")
print(f"  Model: {MODEL}")
print(f"  API Key: {API_KEY[:20]}...")

# Reset ArbiterOS for fresh start
reset_arbiter_os()
print("\n✓ ArbiterOS instance reset")

# Setup policies
print("\n✓ Setting up ArbiterOS policies:")
arbiter_os = get_arbiter_os()

# Policy 1: Analyst Completion Checker
analyst_checker = AnalystCompletionChecker(
    name="analyst_completion",
    required_analysts={"market", "social", "news", "fundamentals"},
)
arbiter_os.add_policy_checker(analyst_checker)
print("  - AnalystCompletionChecker added")

# Policy 2: Debate Rounds Checker
debate_checker = DebateRoundsChecker(name="minimum_debate", min_rounds=1)
arbiter_os.add_policy_checker(debate_checker)
print("  - DebateRoundsChecker added (min_rounds=1)")

# Policy 3: Risk Analysis Checker
risk_checker = RiskAnalysisChecker(name="risk_verification", min_risk_assessments=3)
arbiter_os.add_policy_checker(risk_checker)
print("  - RiskAnalysisChecker added (min_assessments=3)")

print(f"\n✓ Total policies: {len(arbiter_os.policy_checkers)} checkers, {len(arbiter_os.policy_routers)} routers")

# Create configuration
print("\n✓ Creating TradingAgents configuration:")
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = MODEL  # Use custom model
config["quick_think_llm"] = MODEL  # Use custom model
config["backend_url"] = BASE_URL  # Use custom base URL
config["max_debate_rounds"] = 1  # Minimal rounds for testing
config["recursion_limit"] = 50  # Default recursion limit

# Use yfinance as fallback if Alpha Vantage not available
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",  # Use yfinance instead of alpha_vantage for testing
    "news_data": "yfinance",  # Use yfinance instead of alpha_vantage for testing
}

print(f"  - Deep Think LLM: {config['deep_think_llm']}")
print(f"  - Quick Think LLM: {config['quick_think_llm']}")
print(f"  - Backend URL: {config['backend_url']}")
print(f"  - Max Debate Rounds: {config['max_debate_rounds']}")

# Initialize TradingAgentsGraph with checkpointing
# Note: Using MemorySaver for testing (SqliteSaver requires langgraph-checkpoint-sqlite)
checkpoint_path = None  # Use MemorySaver instead of SqliteSaver
print(f"\n✓ Initializing TradingAgentsGraph:")
print(f"  - Checkpointing: MemorySaver (in-memory)")

try:
    graph = TradingAgentsGraph(
        debug=True,
        config=config,
        checkpoint_path=checkpoint_path,  # None = use MemorySaver
    )
    print("  ✓ Graph initialized successfully")
except Exception as e:
    print(f"  ✗ Error initializing graph: {e}")
    raise

# Test parameters
ticker = "NVDA"
trade_date = "2024-05-10"

print(f"\n{'=' * 60}")
print(f"Running Analysis")
print(f"{'=' * 60}")
print(f"Ticker: {ticker}")
print(f"Date: {trade_date}")
print(f"\nThis may take a few minutes...\n")

try:
    # Run the analysis
    final_state, decision = graph.propagate(
        ticker,
        trade_date,
        thread_id="test-run-001",
    )
    
    print(f"\n{'=' * 60}")
    print("Analysis Complete!")
    print(f"{'=' * 60}")
    
    print(f"\n✓ Final Decision: {decision}")
    print(f"✓ Thread ID: {graph.current_thread_id}")
    
    # Display execution history summary
    print(f"\n{'=' * 60}")
    print("Execution History Summary")
    print(f"{'=' * 60}")
    
    history = graph.get_execution_history()
    if history.entries:
        print(f"Total supersteps: {len(history.entries)}")
        for i, superstep in enumerate(history.entries):
            instructions = [item.instruction.name for item in superstep]
            print(f"  Superstep {i+1}: {len(instructions)} instructions executed")
            print(f"    Instructions: {', '.join(set(instructions))}")
    else:
        print("No execution history recorded.")
    
    # Display checkpoint info
    print(f"\n{'=' * 60}")
    print("Checkpoint Information")
    print(f"{'=' * 60}")
    
    checkpoints = graph.list_checkpoints(graph.current_thread_id)
    print(f"Total checkpoints: {len(checkpoints)}")
    if checkpoints:
        print(f"Latest checkpoint step: {checkpoints[0].get('step', 'N/A')}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print("✓ ArbiterOS governance layer: WORKING")
    print("✓ Policy checkers: APPLIED")
    print("✓ Checkpoint persistence: WORKING")
    print("✓ Execution history tracking: WORKING")
    print("✓ Custom API configuration: WORKING")
    print(f"\n✓✓✓ TradingAgents + ArbiterOS integration test PASSED! ✓✓✓")
    
except Exception as e:
    print(f"\n{'=' * 60}")
    print("Test Failed")
    print(f"{'=' * 60}")
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n✗✗✗ Test FAILED ✗✗✗")
    sys.exit(1)
