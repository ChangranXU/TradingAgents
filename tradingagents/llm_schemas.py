"""Pydantic schemas for standardized LLM I/O in TradingAgents.

These schemas enforce the "strict schema + post-call validation" pattern described
in ArbiterOS ACF docs. They are intentionally minimal and designed to preserve
existing downstream behavior by allowing callers to render a string equivalent to
the previous free-form model output.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


class TradeDecision(str, Enum):
    """Allowed trading decisions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DebateArgumentOutput(BaseModel):
    """Structured output for debate-style agent turns."""

    argument: str = Field(
        ...,
        min_length=1,
        description="The debater's argument text, conversational, no special formatting.",
    )


class AnalystReportOutput(BaseModel):
    """Structured output for analyst reports (Markdown allowed inside field)."""

    report_markdown: str = Field(
        ...,
        min_length=1,
        description="The full analyst report in Markdown format.",
    )


class ResearchManagerOutput(BaseModel):
    """Structured output for the portfolio manager decision + plan."""

    recommendation: TradeDecision = Field(
        ...,
        description="Decisive stance derived from the debate (BUY/SELL/HOLD).",
    )
    summary_bull: str = Field(
        ...,
        min_length=1,
        description="Concise summary of strongest bull-side points.",
    )
    summary_bear: str = Field(
        ...,
        min_length=1,
        description="Concise summary of strongest bear-side points.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Explanation for why the recommendation follows from evidence.",
    )
    strategic_actions: List[str] = Field(
        default_factory=list,
        description="Concrete steps to implement the recommendation.",
    )

    def render_plan_text(self) -> str:
        """Render a human-readable plan consistent with prior behavior."""
        actions = (
            "\n".join([f"- {a}" for a in self.strategic_actions])
            if self.strategic_actions
            else "- (none provided)"
        )
        return (
            f"Recommendation: {self.recommendation.value}\n"
            f"Bull Summary: {self.summary_bull}\n"
            f"Bear Summary: {self.summary_bear}\n"
            f"Rationale: {self.rationale}\n"
            f"Strategic Actions:\n{actions}"
        )


class RiskManagerOutput(BaseModel):
    """Structured output for the risk management judge."""

    recommendation: TradeDecision = Field(
        ...,
        description="Final risk-adjusted recommendation (BUY/SELL/HOLD).",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Risk-focused reasoning grounded in the debate and past reflections.",
    )
    refined_trader_plan: str = Field(
        ...,
        min_length=1,
        description="Refined trader plan incorporating risk insights.",
    )

    def render_decision_text(self) -> str:
        """Render a readable decision string similar to prior behavior."""
        return (
            f"Recommendation: {self.recommendation.value}\n\n"
            f"Reasoning:\n{self.reasoning}\n\n"
            f"Refined Plan:\n{self.refined_trader_plan}"
        )


class TraderOutput(BaseModel):
    """Structured output for trader recommendation."""

    decision: TradeDecision = Field(..., description="BUY/SELL/HOLD decision.")
    rationale: str = Field(
        ...,
        min_length=1,
        description="Justification for the decision, referencing the provided context.",
    )

    def render_trader_text(self) -> str:
        """Render legacy trader content with required final line."""
        return (
            f"{self.rationale}\n\n"
            f"FINAL TRANSACTION PROPOSAL: **{self.decision.value}**"
        )


class SignalExtractionOutput(BaseModel):
    """Structured output for decision extraction from free-form text."""

    decision: TradeDecision = Field(..., description="Extracted BUY/SELL/HOLD.")


class ReflectionOutput(BaseModel):
    """Structured output for reflection that is stored into long-term memory."""

    reasoning: str = Field(..., min_length=1, description="Detailed reasoning analysis.")
    improvement: str = Field(
        ...,
        min_length=1,
        description="Corrective actions and improvements for future decisions.",
    )
    summary: str = Field(..., min_length=1, description="Lessons learned summary.")
    query: str = Field(
        ...,
        min_length=1,
        description="Condensed <=1000-token query sentence capturing the essence.",
    )

    def render_reflection_text(self) -> str:
        """Render a readable reflection string similar to prior output."""
        return (
            "Reasoning:\n"
            f"{self.reasoning}\n\n"
            "Improvement:\n"
            f"{self.improvement}\n\n"
            "Summary:\n"
            f"{self.summary}\n\n"
            "Query:\n"
            f"{self.query}"
        )

