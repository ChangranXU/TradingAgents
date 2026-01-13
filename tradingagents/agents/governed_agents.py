"""ArbiterOS governance layer for TradingAgents.

This module provides the ArbiterOS instance and decorator utilities for wrapping
TradingAgents node functions with policy-driven governance.

The governance layer maps each agent role to an appropriate InstructionType:
    - Analysts → GENERATE (generate analysis reports)
    - Tool calls → TOOL_CALL (external data retrieval)
    - Researchers → REFLECT (critique and debate)
    - Research Manager → EVALUATE_PROGRESS (synthesize debate)
    - Trader → DECOMPOSE (break into action plan)
    - Risk Debaters → NEGOTIATE (multi-turn dialogue)
    - Risk Manager → VERIFY (validate trading decisions)
"""

import functools
import logging
from typing import Any, Callable, TypeVar

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.instructions import (
    CognitiveCore,
    ExecutionCore,
    MetacognitiveCore,
    NormativeCore,
    SocialCore,
)

logger = logging.getLogger(__name__)

# Global ArbiterOS instance for the TradingAgents framework
arbiter_os = ArbiterOSAlpha(backend="langgraph")

# Instruction type mappings for different agent roles
ANALYST_INSTRUCTION = CognitiveCore.GENERATE
TOOL_CALL_INSTRUCTION = ExecutionCore.TOOL_CALL
RESEARCHER_INSTRUCTION = CognitiveCore.REFLECT
RESEARCH_MANAGER_INSTRUCTION = MetacognitiveCore.EVALUATE_PROGRESS
TRADER_INSTRUCTION = CognitiveCore.DECOMPOSE
RISK_DEBATER_INSTRUCTION = SocialCore.NEGOTIATE
RISK_MANAGER_INSTRUCTION = NormativeCore.VERIFY

# Type variable for function decoration
F = TypeVar("F", bound=Callable[..., Any])


def govern_analyst(func: F) -> F:
    """Decorator wrapper for analyst node functions.

    Applies GENERATE instruction type governance to analyst functions
    that produce analysis reports from market data.

    Args:
        func: The analyst node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(ANALYST_INSTRUCTION)(func)


def govern_tool_call(func: F) -> F:
    """Decorator wrapper for tool call functions.

    Applies TOOL_CALL instruction type governance to functions
    that interact with external data sources.

    Args:
        func: The tool call function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(TOOL_CALL_INSTRUCTION)(func)


def govern_researcher(func: F) -> F:
    """Decorator wrapper for researcher node functions.

    Applies REFLECT instruction type governance to researcher functions
    that engage in bull/bear debates.

    Args:
        func: The researcher node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(RESEARCHER_INSTRUCTION)(func)


def govern_research_manager(func: F) -> F:
    """Decorator wrapper for research manager node functions.

    Applies EVALUATE_PROGRESS instruction type governance to the
    research manager that synthesizes debate outcomes.

    Args:
        func: The research manager node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(RESEARCH_MANAGER_INSTRUCTION)(func)


def govern_trader(func: F) -> F:
    """Decorator wrapper for trader node functions.

    Applies DECOMPOSE instruction type governance to trader functions
    that break down investment plans into actionable decisions.

    Args:
        func: The trader node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(TRADER_INSTRUCTION)(func)


def govern_risk_debater(func: F) -> F:
    """Decorator wrapper for risk debater node functions.

    Applies NEGOTIATE instruction type governance to risk debater
    functions (risky, safe, neutral analysts).

    Args:
        func: The risk debater node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(RISK_DEBATER_INSTRUCTION)(func)


def govern_risk_manager(func: F) -> F:
    """Decorator wrapper for risk manager node functions.

    Applies VERIFY instruction type governance to the risk manager
    that validates final trading decisions.

    Args:
        func: The risk manager node function to wrap.

    Returns:
        The wrapped function with ArbiterOS governance.
    """
    return arbiter_os.instruction(RISK_MANAGER_INSTRUCTION)(func)


def wrap_factory_result(wrapper_func: Callable[[F], F]) -> Callable:
    """Higher-order decorator for wrapping factory function results.

    This decorator is used to wrap the inner function returned by agent
    factory functions (e.g., create_market_analyst) without modifying
    the factory function itself.

    Args:
        wrapper_func: The governance wrapper to apply (e.g., govern_analyst).

    Returns:
        A decorator that wraps the result of the factory function.

    Example:
        >>> @wrap_factory_result(govern_analyst)
        ... def create_market_analyst(llm):
        ...     def market_analyst_node(state):
        ...         # Original logic unchanged
        ...         return {"market_report": "..."}
        ...     return market_analyst_node
    """

    def factory_decorator(factory_func: Callable) -> Callable:
        @functools.wraps(factory_func)
        def wrapper(*args, **kwargs):
            # Call the original factory to get the node function
            node_func = factory_func(*args, **kwargs)
            # Wrap the node function with governance
            governed_func = wrapper_func(node_func)
            return governed_func

        return wrapper

    return factory_decorator


def get_arbiter_os() -> ArbiterOSAlpha:
    """Get the global ArbiterOS instance for TradingAgents.

    Returns:
        The configured ArbiterOSAlpha instance.
    """
    return arbiter_os


def reset_arbiter_os() -> None:
    """Reset the global ArbiterOS instance.

    Creates a new ArbiterOS instance, clearing all history and policies.
    Useful for testing or when starting fresh executions.
    """
    global arbiter_os
    arbiter_os = ArbiterOSAlpha(backend="langgraph")
    logger.info("ArbiterOS instance reset")
