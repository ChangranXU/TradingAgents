"""Utilities for schema-first LLM invocation with deterministic validation.

This module implements an ACF-style pattern:
- Define an explicit output schema (Pydantic v2 model)
- Force the LLM to emit JSON only
- Parse/validate deterministically
- Retry with error feedback if invalid
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Type, TypeVar, Union, overload

from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


class LlmSchemaError(RuntimeError):
    """Raised when an LLM response cannot be validated against a schema."""


def _schema_instruction(schema: Mapping[str, Any]) -> str:
    """Build a compact instruction to enforce JSON output matching schema.

    Args:
        schema: JSON schema produced from Pydantic.

    Returns:
        A system prompt segment describing formatting constraints.
    """
    schema_text = json.dumps(schema, ensure_ascii=False)
    return (
        "You must respond with a single JSON object and nothing else.\n"
        "Do not wrap in Markdown code fences.\n"
        "Do not include extra keys beyond the schema.\n"
        "Do not include comments.\n"
        f"Output must conform to this JSON Schema:\n{schema_text}"
    )


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from text.

    This is a defensive helper in case the model includes stray text. We still
    validate strictly after extraction.

    Args:
        text: Raw LLM output text.

    Returns:
        A substring that should represent a JSON object.

    Raises:
        ValueError: If no JSON object can be located.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found.")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("Unterminated JSON object in output.")


def parse_validated_json(raw_text: str, output_model: Type[TModel]) -> TModel:
    """Parse and validate an LLM output string as JSON against a Pydantic model.

    Args:
        raw_text: Raw LLM output text which should contain a JSON object.
        output_model: Pydantic model class describing required JSON output.

    Returns:
        Parsed and validated Pydantic model instance.

    Raises:
        LlmSchemaError: If JSON parsing or schema validation fails.
    """
    try:
        json_text = _extract_json_object(raw_text)
        data = json.loads(json_text)
    except Exception as e:  # noqa: BLE001
        raise LlmSchemaError(f"Invalid JSON: {e}") from e
    try:
        return output_model.model_validate(data)
    except ValidationError as e:
        raise LlmSchemaError(f"Schema validation failed: {e}") from e


def _to_messages(
    prompt_or_messages: Union[str, Sequence[Any]],
    extra_system: str,
) -> list[Any]:
    """Normalize input into a message list and prepend/merge system instruction."""
    if isinstance(prompt_or_messages, str):
        return [("system", extra_system), ("human", prompt_or_messages)]

    # If it's already messages, try to merge into first system message if present.
    messages = list(prompt_or_messages)
    if not messages:
        return [("system", extra_system)]

    first = messages[0]
    # Support both ("role", "content") tuples and {"role": ..., "content": ...} dicts
    if isinstance(first, tuple) and len(first) == 2 and first[0] == "system":
        messages[0] = ("system", f"{first[1]}\n\n{extra_system}")
        return messages
    if isinstance(first, Mapping) and first.get("role") == "system":
        merged = dict(first)
        merged["content"] = f"{first.get('content','')}\n\n{extra_system}"
        messages[0] = merged
        return messages

    # Otherwise prepend a system message.
    return [("system", extra_system), *messages]


@dataclass(frozen=True)
class ValidatedLlmResult:
    """Validated model plus raw LLM text for debugging."""

    parsed: BaseModel
    raw_text: str


def invoke_validated_json(
    llm: Any,
    prompt_or_messages: Union[str, Sequence[Any]],
    output_model: Type[TModel],
    *,
    max_retries: int = 2,
) -> ValidatedLlmResult:
    """Invoke an LLM and deterministically validate output against a Pydantic model.

    Args:
        llm: LangChain chat model with `.invoke(...)`.
        prompt_or_messages: Either a prompt string or a LangChain-compatible
            messages sequence.
        output_model: Pydantic model class describing required JSON output.
        max_retries: Number of retries after a schema/JSON failure.

    Returns:
        ValidatedLlmResult containing the parsed model instance and raw text.

    Raises:
        LlmSchemaError: If output cannot be validated after retries.
    """
    schema = output_model.model_json_schema()
    system = _schema_instruction(schema)

    last_err: str | None = None
    cur_input: Union[str, Sequence[Any]] = prompt_or_messages

    for attempt in range(max_retries + 1):
        messages = _to_messages(cur_input, system)
        response = llm.invoke(messages)
        raw_text = getattr(response, "content", None) or str(response)

        try:
            json_text = _extract_json_object(raw_text)
            data = json.loads(json_text)
        except Exception as e:  # noqa: BLE001 - caller sees structured error
            last_err = f"Invalid JSON: {e}"
        else:
            try:
                parsed = output_model.model_validate(data)
                return ValidatedLlmResult(parsed=parsed, raw_text=raw_text)
            except ValidationError as e:
                last_err = f"Schema validation failed: {e}"

        if attempt < max_retries:
            # Feed back the error deterministically and ask for corrected JSON only.
            cur_input = [
                ("system", system),
                ("human", "Your previous output was invalid."),
                ("human", f"Error:\n{last_err}"),
                ("human", "Return corrected JSON only. No extra text."),
                ("human", "Original request follows:"),
                ("human", prompt_or_messages if isinstance(prompt_or_messages, str) else json.dumps(prompt_or_messages, default=str)),
            ]

    raise LlmSchemaError(last_err or "Unknown schema validation error.")

