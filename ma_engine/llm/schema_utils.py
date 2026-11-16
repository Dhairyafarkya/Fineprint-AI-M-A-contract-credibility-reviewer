"""Turn Pydantic models into self-contained JSON schemas for Gemini structured output."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

# JSON Schema keywords kept for Gemini. Numeric bounds and defaults are dropped (Gemini's docs list
# them among causes of "schema too complex" rejections); Pydantic still enforces them on the result.
_ALLOWED = {"type", "description", "enum", "format", "items", "properties", "required", "anyOf"}


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            resolved = _inline_refs(copy.deepcopy(defs[name]), defs)
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolved, **extra}
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _clean(node: Any, in_properties: bool = False) -> Any:
    if isinstance(node, list):
        return [_clean(v) for v in node]
    if not isinstance(node, dict):
        return node
    if in_properties:  # keys are field names, values are schemas
        return {k: _clean(v) for k, v in node.items()}
    out: dict[str, Any] = {}
    if "const" in node:
        out["enum"] = [node["const"]]
    for k, v in node.items():
        if k not in _ALLOWED:
            continue
        if k == "properties":
            out[k] = _clean(v, in_properties=True)
        elif k in ("items", "anyOf", "additionalProperties"):
            out[k] = _clean(v)
        else:
            out[k] = v
    if "enum" in out and "type" not in out:
        out["type"] = "string"
    # Optional[X] -> {"type": [X, "null"]}: simpler for the decoder than anyOf.
    alts = out.get("anyOf")
    if alts and len(alts) == 2 and {"type": "null"} in alts:
        other = next(a for a in alts if a != {"type": "null"})
        if isinstance(other.get("type"), str) and set(other) <= {"type", "description", "enum", "items", "format"}:
            del out["anyOf"]
            out = {**other, **out, "type": [other["type"], "null"]}
    # Every property is required (nullable where optional): avoids long runs of optional
    # properties, which Gemini may reject, and makes the model state "null" explicitly.
    if isinstance(out.get("properties"), dict):
        out["required"] = list(out["properties"])
    return out


def gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with $refs inlined and only Gemini-supported keywords kept."""
    raw = model.model_json_schema()
    schema = _clean(_inline_refs(raw, raw.get("$defs", {})))
    schema["type"] = "object"
    return schema


def schema_name(model: type[BaseModel]) -> str:
    """Stable snake_case name, e.g. ExecutiveSummary -> executive_summary."""
    return "".join("_" + c.lower() if c.isupper() else c for c in model.__name__).lstrip("_")
