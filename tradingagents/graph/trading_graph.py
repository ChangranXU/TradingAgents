# TradingAgents/graph/trading_graph.py

import os
import uuid
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional, Iterator

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.agents.governed_agents import get_arbiter_os
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework with ArbiterOS governance.

    This class provides a governance layer around the TradingAgents multi-agent
    trading framework, enabling policy-based validation, execution history tracking,
    and LangGraph checkpoint persistence for workflow state.

    Attributes:
        checkpointer: The LangGraph checkpointer for state persistence.
        arbiter_os: Reference to the ArbiterOS governance instance.
    """

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        checkpoint_path: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """Initialize the trading agents graph with ArbiterOS governance.

        Args:
            selected_analysts: List of analyst types to include.
                Options: "market", "social", "news", "fundamentals"
            debug: Whether to run in debug mode with verbose tracing.
            config: Configuration dictionary. If None, uses default config.
            checkpoint_path: Path to SQLite database for checkpointing.
                Defaults to "./checkpoints/trading.db" if not provided.
                Ignored if checkpointer is provided.
            checkpointer: Optional custom LangGraph checkpointer.
                If provided, takes precedence over checkpoint_path.
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.selected_analysts = selected_analysts

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize checkpointer for state persistence
        if checkpointer is not None:
            self.checkpointer = checkpointer
        elif checkpoint_path is None:
            # Use in-memory checkpointer when no path specified
            self.checkpointer = MemorySaver()
        else:
            # Use SQLite checkpointer if available
            if SqliteSaver is None:
                raise ImportError(
                    "SqliteSaver requires 'langgraph-checkpoint-sqlite' package. "
                    "Install it with: pip install langgraph-checkpoint-sqlite"
                )
            checkpoint_dir = Path(checkpoint_path).parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpointer = SqliteSaver.from_conn_string(checkpoint_path)

        # Get reference to ArbiterOS instance
        self.arbiter_os = get_arbiter_os()

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph with checkpointer
        self.graph = self.graph_setup.setup_graph(
            selected_analysts, checkpointer=self.checkpointer
        )

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(
        self,
        company_name: str,
        trade_date: str,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Run the trading agents graph for a company on a specific date.

        Args:
            company_name: The ticker symbol or company name to analyze.
            trade_date: The date for which to make trading decisions.
            thread_id: Optional thread ID for checkpoint tracking.
                If not provided, a new UUID will be generated.
            resume: If True, attempts to resume from the last checkpoint
                for the given thread_id. If False, starts fresh.

        Returns:
            A tuple of (final_state, processed_signal) where:
                - final_state: The complete workflow state after execution.
                - processed_signal: The extracted trading decision (BUY/SELL/HOLD).
        """
        self.ticker = company_name

        # Generate or use provided thread_id
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        self.current_thread_id = thread_id

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )

        # Configure checkpoint tracking
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.get("recursion_limit", 100),
        }

        # Check if resuming from checkpoint
        if resume:
            saved_state = self.get_state(thread_id)
            if saved_state:
                init_agent_state = saved_state

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, config=config, stream_mode="values"):
                if "messages" in chunk and len(chunk["messages"]) > 0:
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)

            final_state = trace[-1] if trace else init_agent_state
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, config=config)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    # ==================== Checkpoint Management ====================

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the saved state for a given thread_id.

        Args:
            thread_id: The thread identifier for the checkpoint.

        Returns:
            The saved state dictionary, or None if no checkpoint exists.
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_snapshot = self.graph.get_state(config)
            if state_snapshot and state_snapshot.values:
                return state_snapshot.values
        except Exception:
            pass
        return None

    def update_state(
        self,
        thread_id: str,
        values: Dict[str, Any],
        as_node: Optional[str] = None,
    ) -> None:
        """Update the saved state for a given thread_id.

        This allows manual intervention in the workflow state, useful for
        correcting errors or adjusting trading parameters mid-execution.

        Args:
            thread_id: The thread identifier for the checkpoint.
            values: Dictionary of state values to update.
            as_node: Optional node name to attribute the update to.
        """
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, values, as_node=as_node)

    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a given thread_id.

        Args:
            thread_id: The thread identifier to list checkpoints for.

        Returns:
            List of checkpoint metadata dictionaries, each containing:
                - checkpoint_id: Unique checkpoint identifier
                - timestamp: When the checkpoint was created
                - step: The step number in the workflow
        """
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []
        try:
            for checkpoint in self.graph.get_state_history(config):
                checkpoints.append({
                    "checkpoint_id": checkpoint.config.get("configurable", {}).get(
                        "checkpoint_id"
                    ),
                    "timestamp": checkpoint.metadata.get("created_at") if checkpoint.metadata else None,
                    "step": checkpoint.metadata.get("step") if checkpoint.metadata else None,
                    "next_nodes": list(checkpoint.next) if checkpoint.next else [],
                })
        except Exception:
            pass
        return checkpoints

    def get_execution_history(self) -> "History":
        """Get the ArbiterOS execution history for governance tracking.

        Returns:
            The History object containing all tracked instruction executions.
        """
        return self.arbiter_os.history

    def print_execution_history(self) -> None:
        """Pretty-print the ArbiterOS execution history."""
        self.arbiter_os.history.pprint()
