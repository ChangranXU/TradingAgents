"""Trading-specific policy checkers for ArbiterOS governance.

This module provides policy checkers designed for the TradingAgents
multi-agent trading framework, enforcing trading workflow constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Set

from arbiteros_alpha.history import History
from arbiteros_alpha.instructions import CognitiveCore, MetacognitiveCore, SocialCore
from arbiteros_alpha.policy import PolicyChecker

logger = logging.getLogger(__name__)


@dataclass
class AnalystCompletionChecker(PolicyChecker):
    """Policy checker that verifies all selected analysts have completed.

    This checker ensures that all configured analyst types have produced
    their reports before researchers begin their debate phase.

    Attributes:
        name: Human-readable name for this policy checker.
        required_analysts: Set of analyst types that must complete.
            Valid values: "market", "social", "news", "fundamentals".
        completed_analysts: Tracks which analysts have completed (internal).

    Example:
        >>> checker = AnalystCompletionChecker(
        ...     name="all_analysts_complete",
        ...     required_analysts={"market", "news", "fundamentals"}
        ... )
        >>> arbiter_os.add_policy_checker(checker)
    """

    name: str
    required_analysts: Set[str] = field(
        default_factory=lambda: {"market", "social", "news", "fundamentals"}
    )
    _completed_count: int = field(default=0, init=False)

    def check_before(self, history: History) -> bool:
        """Check if all required analysts have completed their reports.

        Counts the number of GENERATE instructions executed (analysts)
        and verifies it meets or exceeds the required analyst count.

        Args:
            history: The execution history up to this point.

        Returns:
            True if enough analysts have completed or if we're still
            in the analyst phase. Returns False only if researchers
            start before analysts complete.
        """
        if not history.entries:
            return True

        # Count GENERATE instructions (analysts)
        generate_count = 0
        reflect_started = False

        for superstep in history.entries:
            for item in superstep:
                if item.instruction == CognitiveCore.GENERATE:
                    generate_count += 1
                elif item.instruction == CognitiveCore.REFLECT:
                    reflect_started = True
                    break
            if reflect_started:
                break

        # If researchers (REFLECT) have started, verify analysts completed
        if reflect_started and generate_count < len(self.required_analysts):
            logger.error(
                f"AnalystCompletionChecker '{self.name}': Only {generate_count} "
                f"analysts completed, but {len(self.required_analysts)} required"
            )
            return False

        return True


@dataclass
class DebateRoundsChecker(PolicyChecker):
    """Policy checker that ensures minimum debate rounds occurred.

    This checker enforces that bull/bear researchers engage in at least
    a minimum number of debate rounds before the research manager makes
    a decision.

    Attributes:
        name: Human-readable name for this policy checker.
        min_rounds: Minimum number of debate rounds required.

    Example:
        >>> checker = DebateRoundsChecker(
        ...     name="minimum_debate",
        ...     min_rounds=2
        ... )
        >>> arbiter_os.add_policy_checker(checker)
    """

    name: str
    min_rounds: int = 1

    def check_before(self, history: History) -> bool:
        """Check if minimum debate rounds have occurred.

        Counts REFLECT instructions (researcher debates) and verifies
        the count meets the minimum before EVALUATE_PROGRESS (manager).

        Args:
            history: The execution history up to this point.

        Returns:
            True if minimum rounds met or not yet at manager stage.
            False if manager starts without enough debate rounds.
        """
        if not history.entries:
            return True

        reflect_count = 0
        evaluate_started = False

        for superstep in history.entries:
            for item in superstep:
                if item.instruction == CognitiveCore.REFLECT:
                    reflect_count += 1
                elif item.instruction == MetacognitiveCore.EVALUATE_PROGRESS:
                    evaluate_started = True
                    break
            if evaluate_started:
                break

        # Check if manager started before minimum debate rounds
        if evaluate_started and reflect_count < (self.min_rounds * 2):
            # *2 because each round has bull + bear
            logger.error(
                f"DebateRoundsChecker '{self.name}': Only {reflect_count // 2} "
                f"debate rounds completed, but {self.min_rounds} required"
            )
            return False

        return True


@dataclass
class RiskAnalysisChecker(PolicyChecker):
    """Policy checker that verifies risk team analysis before final trade.

    This checker ensures that all three risk analysts (risky, safe, neutral)
    have provided their assessments before the risk manager makes the final
    trading decision.

    Attributes:
        name: Human-readable name for this policy checker.
        min_risk_assessments: Minimum number of risk assessments required.

    Example:
        >>> checker = RiskAnalysisChecker(
        ...     name="risk_consensus",
        ...     min_risk_assessments=3
        ... )
        >>> arbiter_os.add_policy_checker(checker)
    """

    name: str
    min_risk_assessments: int = 3

    def check_before(self, history: History) -> bool:
        """Check if risk team has completed analysis.

        Counts NEGOTIATE instructions (risk debaters) and verifies
        the count meets minimum before VERIFY (risk manager).

        Args:
            history: The execution history up to this point.

        Returns:
            True if enough risk assessments completed or not at final stage.
            False if risk manager decides without sufficient analysis.
        """
        if not history.entries:
            return True

        negotiate_count = 0
        verify_started = False

        for superstep in history.entries:
            for item in superstep:
                if item.instruction == SocialCore.NEGOTIATE:
                    negotiate_count += 1
                elif hasattr(item.instruction, "name") and item.instruction.name == "VERIFY":
                    verify_started = True
                    break
            if verify_started:
                break

        # Check if risk manager started before minimum assessments
        if verify_started and negotiate_count < self.min_risk_assessments:
            logger.error(
                f"RiskAnalysisChecker '{self.name}': Only {negotiate_count} "
                f"risk assessments completed, but {self.min_risk_assessments} required"
            )
            return False

        return True
