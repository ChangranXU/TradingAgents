"""Trading-specific policy routers for ArbiterOS governance.

This module provides policy routers designed for the TradingAgents
multi-agent trading framework, enabling dynamic workflow routing.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from arbiteros_alpha.history import History
from arbiteros_alpha.policy import PolicyRouter

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceRouter(PolicyRouter):
    """Policy router that redirects based on trading confidence levels.

    This router monitors the confidence expressed in trading decisions
    and routes back to analysts for re-evaluation if confidence is too low.

    Attributes:
        name: Human-readable name for this policy router.
        threshold: Minimum confidence score (0.0-1.0) to proceed.
        target: The node name to route to when confidence is below threshold.
        confidence_key: The key in output state containing confidence score.

    Example:
        >>> router = ConfidenceRouter(
        ...     name="low_confidence_review",
        ...     threshold=0.7,
        ...     target="Market Analyst"
        ... )
        >>> arbiter_os.add_policy_router(router)
    """

    name: str
    threshold: float = 0.7
    target: str = "Market Analyst"
    confidence_key: str = "confidence"

    def route_after(self, history: History) -> Optional[str]:
        """Route to target node if confidence is below threshold.

        Extracts the confidence score from the most recent instruction's
        output state and compares it against the configured threshold.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The target node name if confidence < threshold, None otherwise.
        """
        if not history.entries or not history.entries[-1]:
            return None

        last_entry = history.entries[-1][-1]
        output = last_entry.output_state or {}

        confidence = output.get(self.confidence_key)
        if confidence is not None and confidence < self.threshold:
            logger.warning(
                f"ConfidenceRouter '{self.name}': Confidence {confidence:.2f} "
                f"below threshold {self.threshold}, routing to {self.target}"
            )
            return self.target

        return None


@dataclass
class RiskOverrideRouter(PolicyRouter):
    """Policy router that escalates high-risk trades for additional review.

    This router monitors risk assessments and routes high-risk decisions
    to an additional review node when the risk score exceeds a threshold.

    Attributes:
        name: Human-readable name for this policy router.
        max_risk_score: Maximum acceptable risk score (0.0-1.0).
        target: The node name to route to for additional review.
        risk_patterns: Keywords indicating high risk in text output.

    Example:
        >>> router = RiskOverrideRouter(
        ...     name="high_risk_escalation",
        ...     max_risk_score=0.8,
        ...     target="Safe Analyst"
        ... )
        >>> arbiter_os.add_policy_router(router)
    """

    name: str
    max_risk_score: float = 0.8
    target: str = "Safe Analyst"
    risk_patterns: tuple = (
        "high risk",
        "extremely volatile",
        "significant downside",
        "major concern",
        "strongly advise against",
    )

    def route_after(self, history: History) -> Optional[str]:
        """Route to target node if risk assessment indicates high risk.

        Analyzes the output for risk indicators and routes to additional
        review if high-risk patterns are detected.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The target node name if high risk detected, None otherwise.
        """
        if not history.entries or not history.entries[-1]:
            return None

        last_entry = history.entries[-1][-1]
        output = last_entry.output_state or {}

        # Check for explicit risk score
        risk_score = output.get("risk_score")
        if risk_score is not None and risk_score > self.max_risk_score:
            logger.warning(
                f"RiskOverrideRouter '{self.name}': Risk score {risk_score:.2f} "
                f"exceeds max {self.max_risk_score}, routing to {self.target}"
            )
            return self.target

        # Check for risk patterns in text content
        for key in ["trader_investment_plan", "investment_plan", "final_trade_decision"]:
            content = output.get(key, "")
            if content:
                content_lower = content.lower()
                for pattern in self.risk_patterns:
                    if pattern in content_lower:
                        logger.warning(
                            f"RiskOverrideRouter '{self.name}': High-risk pattern "
                            f"'{pattern}' detected, routing to {self.target}"
                        )
                        return self.target

        return None


@dataclass
class VolatilityRouter(PolicyRouter):
    """Policy router that adjusts strategy based on market volatility.

    This router monitors market reports for volatility indicators and
    routes to more conservative analysis when high volatility is detected.

    Attributes:
        name: Human-readable name for this policy router.
        target: The node name to route to when high volatility detected.
        volatility_indicators: Keywords indicating high market volatility.
        atr_threshold: ATR percentage threshold for high volatility.

    Example:
        >>> router = VolatilityRouter(
        ...     name="volatility_adjustment",
        ...     target="Safe Analyst"
        ... )
        >>> arbiter_os.add_policy_router(router)
    """

    name: str
    target: str = "Safe Analyst"
    volatility_indicators: tuple = (
        "extreme volatility",
        "market crash",
        "black swan",
        "circuit breaker",
        "flash crash",
        "panic selling",
        "market turbulence",
    )
    atr_threshold: float = 5.0  # ATR percentage indicating high volatility

    def route_after(self, history: History) -> Optional[str]:
        """Route to target node if high market volatility detected.

        Analyzes market reports for volatility indicators and routes
        to more conservative analysis when warranted.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The target node name if high volatility detected, None otherwise.
        """
        if not history.entries or not history.entries[-1]:
            return None

        last_entry = history.entries[-1][-1]
        output = last_entry.output_state or {}

        # Check market report for volatility indicators
        market_report = output.get("market_report", "")
        if market_report:
            report_lower = market_report.lower()
            for indicator in self.volatility_indicators:
                if indicator in report_lower:
                    logger.warning(
                        f"VolatilityRouter '{self.name}': Volatility indicator "
                        f"'{indicator}' detected, routing to {self.target}"
                    )
                    return self.target

            # Check for high ATR values in the report
            atr_match = re.search(r"atr[:\s]*(\d+\.?\d*)%?", report_lower)
            if atr_match:
                atr_value = float(atr_match.group(1))
                if atr_value > self.atr_threshold:
                    logger.warning(
                        f"VolatilityRouter '{self.name}': ATR {atr_value}% exceeds "
                        f"threshold {self.atr_threshold}%, routing to {self.target}"
                    )
                    return self.target

        return None
