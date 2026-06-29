from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from ..localize import LocalizationReport, localize_failures
from ..model import Activity, Journal


MatchMode = Literal["search", "fullmatch"]
MATCH_MODES: tuple[MatchMode, ...] = ("search", "fullmatch")
REGEX_ORACLE_TS = "2026-06-28T00:00:00Z"


@dataclass(frozen=True, slots=True)
class RegexPatternSpec:
    id: str
    pattern: str
    inputs: tuple[str, ...] = ()
    generate: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "inputs": list(self.inputs),
            "generate": self.generate,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class RegexObservation:
    pattern_id: str
    pattern: str
    input_index: int
    input_text: str
    reference: bool
    candidate: bool

    @property
    def agreed(self) -> bool:
        return self.reference == self.candidate

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "pattern": self.pattern,
            "input_index": self.input_index,
            "input_text": self.input_text,
            "reference": self.reference,
            "candidate": self.candidate,
            "agreed": self.agreed,
        }


@dataclass(frozen=True, slots=True)
class RegexPatternResult:
    spec: RegexPatternSpec
    inputs: tuple[str, ...]
    observations: tuple[RegexObservation, ...]
    localization: LocalizationReport
    journal_tail_hash: str
    compile_error: str = ""

    @property
    def supported(self) -> bool:
        return self.compile_error == ""

    @property
    def first_divergence(self) -> RegexObservation | None:
        for observation in self.observations:
            if not observation.agreed:
                return observation
        return None

    @property
    def input_count(self) -> int:
        return len(self.observations)

    @property
    def agreement_count(self) -> int:
        return sum(1 for observation in self.observations if observation.agreed)

    @property
    def divergence_count(self) -> int:
        return self.input_count - self.agreement_count

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "inputs": list(self.inputs),
            "observations": [observation.to_dict() for observation in self.observations],
            "localization": self.localization.to_dict(),
            "journal_tail_hash": self.journal_tail_hash,
            "compile_error": self.compile_error,
        }


@dataclass(frozen=True, slots=True)
class RegexOracleReport:
    pattern_results: tuple[RegexPatternResult, ...]
    reference_mode: MatchMode
    candidate_mode: MatchMode
    max_generated: int

    @property
    def total_inputs(self) -> int:
        return sum(result.input_count for result in self.pattern_results)

    @property
    def agreement_count(self) -> int:
        return sum(result.agreement_count for result in self.pattern_results)

    @property
    def divergence_count(self) -> int:
        return sum(result.divergence_count for result in self.pattern_results)

    @property
    def supported_pattern_count(self) -> int:
        return sum(1 for result in self.pattern_results if result.supported)

    @property
    def unsupported_pattern_count(self) -> int:
        return len(self.pattern_results) - self.supported_pattern_count

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_mode": self.reference_mode,
            "candidate_mode": self.candidate_mode,
            "max_generated": self.max_generated,
            "summary": {
                "patterns": len(self.pattern_results),
                "supported_patterns": self.supported_pattern_count,
                "unsupported_patterns": self.unsupported_pattern_count,
                "inputs": self.total_inputs,
                "agreements": self.agreement_count,
                "divergences": self.divergence_count,
            },
            "pattern_results": [result.to_dict() for result in self.pattern_results],
        }


def load_pattern_set(path: Path) -> tuple[RegexPatternSpec, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_patterns = data if isinstance(data, list) else data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(raw_patterns, list):
        raise ValueError("pattern set must be a list or an object with a patterns list")

    specs: list[RegexPatternSpec] = []
    for index, raw in enumerate(raw_patterns):
        if not isinstance(raw, dict):
            raise ValueError(f"pattern entry {index} must be an object")
        pattern = raw.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError(f"pattern entry {index} is missing a string pattern")
        raw_inputs = raw.get("inputs", [])
        if raw_inputs is None:
            raw_inputs = []
        if not isinstance(raw_inputs, list) or not all(isinstance(item, str) for item in raw_inputs):
            raise ValueError(f"pattern entry {index} inputs must be a list of strings")
        pattern_id = raw.get("id", f"pattern-{index}")
        if not isinstance(pattern_id, str) or not pattern_id:
            raise ValueError(f"pattern entry {index} id must be a non-empty string")
        generate = raw.get("generate", len(raw_inputs) == 0)
        if not isinstance(generate, bool):
            raise ValueError(f"pattern entry {index} generate must be a boolean")
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"pattern entry {index} description must be a string")
        specs.append(
            RegexPatternSpec(
                id=pattern_id,
                pattern=pattern,
                inputs=tuple(raw_inputs),
                generate=generate,
                description=description,
            )
        )
    return tuple(specs)


def run_regex_oracle(
    specs: tuple[RegexPatternSpec, ...] | list[RegexPatternSpec],
    *,
    reference_mode: MatchMode = "search",
    candidate_mode: MatchMode = "fullmatch",
    max_generated: int = 24,
) -> RegexOracleReport:
    _validate_mode("reference_mode", reference_mode)
    _validate_mode("candidate_mode", candidate_mode)
    if max_generated < 0:
        raise ValueError("max_generated must be non-negative")

    results = [
        _run_pattern(
            spec,
            reference_mode=reference_mode,
            candidate_mode=candidate_mode,
            max_generated=max_generated,
        )
        for spec in specs
    ]
    return RegexOracleReport(
        pattern_results=tuple(results),
        reference_mode=reference_mode,
        candidate_mode=candidate_mode,
        max_generated=max_generated,
    )


def generate_inputs(pattern: str, *, max_generated: int = 24) -> tuple[str, ...]:
    if max_generated <= 0:
        return ()

    literal = _literal_fragment(pattern)
    candidates = ["", "a", "b", "0", " ", "x", "xx"]
    if literal:
        candidates.extend(
            [
                literal,
                literal[:1],
                literal[-1:],
                f"x{literal}",
                f"{literal}x",
                f"x{literal}x",
                literal * 2,
            ]
        )

    chars = _unique_chars(literal)
    candidates.extend(chars)
    if len(chars) >= 2:
        candidates.extend([chars[0] + chars[1], chars[1] + chars[0], chars[0] * 2])

    return tuple(_dedupe(candidates)[:max_generated])


def render_cli_report(report: RegexOracleReport) -> str:
    lines = [
        (
            "regex_oracle"
            f"\treference={report.reference_mode}"
            f"\tcandidate={report.candidate_mode}"
            f"\tpatterns={report.supported_pattern_count}/{len(report.pattern_results)}"
            f"\tunsupported={report.unsupported_pattern_count}"
        )
    ]

    for result in report.pattern_results:
        if not result.supported:
            lines.append(
                "unsupported"
                f"\tpattern={_cell(result.spec.id)}"
                f"\terror={_cell(result.compile_error)}"
            )
            continue

        divergence = result.first_divergence
        if divergence is None:
            lines.append(
                "no_divergence"
                f"\tpattern={_cell(result.spec.id)}"
                f"\tinputs={result.input_count}"
                f"\tagreement={result.agreement_count}/{result.input_count}"
            )
        else:
            failure = _localization_for_step(result.localization, divergence.input_index)
            lines.append(
                "first_divergence"
                f"\tpattern={_cell(result.spec.id)}"
                f"\tinput_index={divergence.input_index}"
                f"\tinput={_cell(divergence.input_text)}"
                f"\treference={_bool_cell(divergence.reference)}"
                f"\tcandidate={_bool_cell(divergence.candidate)}"
                f"\tbreaking_step={'' if failure is None else failure.breaking_step}"
                f"\tgate={'' if failure is None else _cell(failure.breaking_gate)}"
                f"\tconfidence={'' if failure is None else _cell(failure.confidence)}"
            )
        lines.append(
            "pattern_summary"
            f"\tpattern={_cell(result.spec.id)}"
            f"\tagree={result.agreement_count}/{result.input_count}"
            f"\tdiverge={result.divergence_count}/{result.input_count}"
        )

    lines.append(
        "agreement_summary"
        f"\tagree={report.agreement_count}/{report.total_inputs}"
        f"\tdiverge={report.divergence_count}/{report.total_inputs}"
        f"\tunsupported={report.unsupported_pattern_count}/{len(report.pattern_results)}"
    )
    return "\n".join(lines) + "\n"


def _run_pattern(
    spec: RegexPatternSpec,
    *,
    reference_mode: MatchMode,
    candidate_mode: MatchMode,
    max_generated: int,
) -> RegexPatternResult:
    inputs = _inputs_for_spec(spec, max_generated=max_generated)
    try:
        compiled = re.compile(spec.pattern)
    except re.error as exc:
        return RegexPatternResult(
            spec=spec,
            inputs=inputs,
            observations=(),
            localization=LocalizationReport(status="unsupported"),
            journal_tail_hash="",
            compile_error=str(exc),
        )

    observations = tuple(
        RegexObservation(
            pattern_id=spec.id,
            pattern=spec.pattern,
            input_index=index,
            input_text=text,
            reference=_matches(compiled, text, reference_mode),
            candidate=_matches(compiled, text, candidate_mode),
        )
        for index, text in enumerate(inputs)
    )
    activities = _activities_for_observations(observations)
    journal = Journal.from_activities(activities)
    tail_hash = journal.records[-1].record_hash if journal.records else ""
    return RegexPatternResult(
        spec=spec,
        inputs=inputs,
        observations=observations,
        localization=localize_failures(activities),
        journal_tail_hash=tail_hash,
    )


def _inputs_for_spec(spec: RegexPatternSpec, *, max_generated: int) -> tuple[str, ...]:
    candidates = list(spec.inputs)
    if spec.generate or not candidates:
        candidates.extend(generate_inputs(spec.pattern, max_generated=max_generated))
    return tuple(_dedupe(candidates))


def _activities_for_observations(observations: tuple[RegexObservation, ...]) -> list[Activity]:
    activities: list[Activity] = []
    for observation in observations:
        activities.append(
            Activity(
                run_id=f"regex-oracle:{observation.pattern_id}",
                step_index=observation.input_index,
                ts=REGEX_ORACLE_TS,
                kind="test_run",
                tool_name="PowerShell",
                inputs={
                    "command": (
                        "agent-bisect regex-oracle"
                        f" --pattern-id {observation.pattern_id}"
                        f" --input-index {observation.input_index}"
                    ),
                    "pattern": observation.pattern,
                    "input_text": observation.input_text,
                },
                outputs={
                    "exit_code": 0 if observation.agreed else 1,
                    "result_text": "1 passed" if observation.agreed else "1 failed",
                    "reference": observation.reference,
                    "candidate": observation.candidate,
                },
                target="shell",
                parent_step=None if observation.input_index == 0 else observation.input_index - 1,
            )
        )
    return activities


def _matches(compiled: re.Pattern[str], text: str, mode: MatchMode) -> bool:
    if mode == "search":
        return compiled.search(text) is not None
    if mode == "fullmatch":
        return compiled.fullmatch(text) is not None
    raise ValueError(f"unsupported match mode: {mode}")


def _validate_mode(name: str, value: str) -> None:
    if value not in MATCH_MODES:
        choices = ", ".join(MATCH_MODES)
        raise ValueError(f"{name} must be one of: {choices}")


def _literal_fragment(pattern: str) -> str:
    literal_chars: list[str] = []
    escaped = False
    for char in pattern:
        if escaped:
            literal_chars.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char.isalnum() or char in {" ", "_", "-"}:
            literal_chars.append(char)
    return "".join(literal_chars[:8])


def _unique_chars(text: str) -> list[str]:
    seen: set[str] = set()
    chars: list[str] = []
    for char in text:
        if char not in seen:
            seen.add(char)
            chars.append(char)
    return chars


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _localization_for_step(report: LocalizationReport, step_index: int) -> Any:
    for failure in report.failures:
        if failure.breaking_step == step_index or step_index in failure.failure_cascade:
            return failure
    return None


def _cell(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _bool_cell(value: bool) -> str:
    return "true" if value else "false"
