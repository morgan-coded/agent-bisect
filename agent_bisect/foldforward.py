from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gates import GateResult
from .model import Activity


@dataclass(slots=True)
class FoldForwardState:
    """Reconstruct only journal-provided file contents for deterministic G2 checks.

    G2 verifies the deterministic envelope of recorded edits. It never reads the
    live on-disk file, because the tree may have drifted and may contain sensitive data.
    Without a prior Write/full-content anchor in the same run, Edit fragments are
    intentionally reported as NA rather than guessed.
    """

    contents: dict[str, str] = field(default_factory=dict)

    def check_activity(self, activity: Activity) -> GateResult:
        if activity.kind != "file_edit":
            return GateResult("G2", "NA", "not a file_edit", activity.step_index)

        file_path = activity.inputs.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return GateResult("G2", "FAIL", "file_edit malformed: missing file_path", activity.step_index)

        if _is_write(activity):
            content = activity.inputs.get("new_string")
            if not isinstance(content, str):
                return GateResult("G2", "FAIL", "write malformed: missing content", activity.step_index)
            self.contents[file_path] = content
            return GateResult("G2", "PASS", "full-content anchor established", activity.step_index)

        edits = _edits_for(activity)
        if not edits:
            return GateResult("G2", "FAIL", "file_edit malformed: missing edit fragments", activity.step_index)

        if file_path not in self.contents:
            return GateResult("G2", "NA", "no full-content anchor", activity.step_index)

        content = self.contents[file_path]
        for edit_index, edit in enumerate(edits, start=1):
            old_string = edit.get("old_string")
            new_string = edit.get("new_string")
            if not isinstance(old_string, str) or not isinstance(new_string, str):
                return GateResult("G2", "FAIL", f"edit {edit_index} malformed", activity.step_index)
            if old_string == "":
                return GateResult("G2", "FAIL", f"edit {edit_index} empty old_string", activity.step_index)

            matches = content.count(old_string)
            if matches == 0:
                return GateResult("G2", "FAIL", f"edit {edit_index} old_string not found", activity.step_index)
            if matches > 1:
                return GateResult("G2", "FAIL", f"edit {edit_index} old_string ambiguous", activity.step_index)
            content = content.replace(old_string, new_string, 1)

        self.contents[file_path] = content
        if len(edits) == 1:
            return GateResult("G2", "PASS", "old_string matched uniquely", activity.step_index)
        return GateResult("G2", "PASS", f"{len(edits)} edits matched uniquely", activity.step_index)


def run_fold_forward(activities: list[Activity]) -> list[GateResult]:
    state = FoldForwardState()
    return [state.check_activity(activity) for activity in activities]


def _is_write(activity: Activity) -> bool:
    return activity.tool_name == "Write" or activity.inputs.get("write_mode") is True


def _edits_for(activity: Activity) -> list[dict[str, Any]]:
    if activity.tool_name == "MultiEdit":
        edits = activity.inputs.get("edits")
        if isinstance(edits, list):
            return [edit for edit in edits if isinstance(edit, dict)]
    return [
        {
            "old_string": activity.inputs.get("old_string"),
            "new_string": activity.inputs.get("new_string"),
        }
    ]
