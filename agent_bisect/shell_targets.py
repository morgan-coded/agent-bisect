from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import shlex
from typing import Any


@dataclass(frozen=True, slots=True)
class ShellTargets:
    reads: list[str]
    writes: list[str]


EMPTY_TARGETS = ShellTargets(reads=[], writes=[])
NO_VALUE_FLAGS = {"-q", "-s", "-v", "-x", "--quiet", "--verbose"}
VALUE_FLAGS = {"-k", "-m", "--maxfail", "--tb"}


def extract_shell_targets(command: Any) -> ShellTargets:
    if not isinstance(command, str) or not command.strip():
        return EMPTY_TARGETS
    if _has_ambiguous_shell_syntax(command):
        return EMPTY_TARGETS

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return EMPTY_TARGETS
    if not tokens:
        return EMPTY_TARGETS

    command_tokens, redirect_writes = _strip_output_redirects(tokens)
    if command_tokens is None:
        return EMPTY_TARGETS

    reads: set[str] = set()
    writes: set[str] = set(redirect_writes)
    if command_tokens:
        command_reads, command_writes = _targets_from_command(command_tokens)
        if command_reads is None or command_writes is None:
            return ShellTargets(reads=sorted(reads), writes=sorted(writes))
        reads.update(command_reads)
        writes.update(command_writes)

    return ShellTargets(reads=sorted(reads), writes=sorted(writes))


def _has_ambiguous_shell_syntax(command: str) -> bool:
    ambiguous_literals = ("|", "&&", "||", "&", ";", "\n", "\r", "$", "`")
    if any(literal in command for literal in ambiguous_literals):
        return True
    if "$(" in command:
        return True
    return any(char in command for char in "*?[{}~")


def _strip_output_redirects(tokens: list[str]) -> tuple[list[str], set[str]] | tuple[None, None]:
    command_tokens: list[str] = []
    writes: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {">", ">>"}:
            if index + 1 >= len(tokens) or not _is_literal_path(tokens[index + 1]):
                return None, None
            writes.add(tokens[index + 1])
            index += 2
            continue
        if token.startswith(">>") and len(token) > 2:
            path = token[2:]
            if not _is_literal_path(path):
                return None, None
            writes.add(path)
            index += 1
            continue
        if token.startswith(">") and len(token) > 1:
            path = token[1:]
            if not _is_literal_path(path):
                return None, None
            writes.add(path)
            index += 1
            continue
        if token == "<" or token.startswith("<"):
            return None, None
        command_tokens.append(token)
        index += 1
    return command_tokens, writes


def _targets_from_command(tokens: list[str]) -> tuple[set[str] | None, set[str] | None]:
    command = _command_name(tokens[0])
    args = tokens[1:]
    if command in {"cat", "less"}:
        return _all_literal_operands(args), set()
    if command in {"head", "tail"}:
        return _head_tail_reads(args), set()
    if command == "tee":
        return set(), _tee_writes(args)
    if command in {"cp", "mv"}:
        return _copy_move_targets(command, args)
    if command == "rm":
        return set(), _rm_writes(args)
    if command in {"touch", "mkdir"}:
        return set(), _touch_mkdir_writes(command, args)
    if command == "sed":
        return _sed_targets(args)
    if command in {"python", "python3", "py"}:
        return _python_reads(args), set()
    if command in {"node"}:
        return _node_reads(args), set()
    if command == "pytest":
        return _test_path_reads(args), set()
    if command == "go" and args[:1] == ["test"]:
        return _test_path_reads(args[1:]), set()
    if command == "npm":
        return _npm_reads(args), set()
    return None, None


def _all_literal_operands(args: list[str]) -> set[str] | None:
    if not args:
        return set()
    if any(arg.startswith("-") for arg in args):
        return None
    refs = set(args)
    if not all(_is_literal_path(ref) for ref in refs):
        return None
    return refs


def _head_tail_reads(args: list[str]) -> set[str] | None:
    refs: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-n":
            if index + 1 >= len(args) or not args[index + 1].isdigit():
                return None
            index += 2
            continue
        if arg.startswith("-n") and arg[2:].isdigit():
            index += 1
            continue
        if arg.startswith("-") and arg[1:].isdigit():
            index += 1
            continue
        refs.append(arg)
        index += 1
    if not refs:
        return set()
    normalized = set(refs)
    if not all(_is_literal_path(ref) for ref in normalized):
        return None
    return normalized


def _tee_writes(args: list[str]) -> set[str] | None:
    refs: list[str] = []
    for arg in args:
        if arg == "-a":
            continue
        if arg.startswith("-"):
            return None
        refs.append(arg)
    if not refs:
        return set()
    if not all(_is_literal_path(ref) for ref in refs):
        return None
    return set(refs)


def _copy_move_targets(command: str, args: list[str]) -> tuple[set[str] | None, set[str] | None]:
    if len(args) != 2 or any(arg.startswith("-") for arg in args):
        return None, None
    source, destination = args
    if not _is_literal_path(source) or not _is_literal_path(destination):
        return None, None
    if command == "mv":
        return {source}, {source, destination}
    return {source}, {destination}


def _rm_writes(args: list[str]) -> set[str] | None:
    refs: list[str] = []
    for arg in args:
        if arg in {"-f", "-r", "-R", "-rf", "-fr", "-Rf", "-fR"}:
            continue
        if arg.startswith("-"):
            return None
        refs.append(arg)
    if not refs:
        return set()
    if not all(_is_literal_path(ref) for ref in refs):
        return None
    return set(refs)


def _touch_mkdir_writes(command: str, args: list[str]) -> set[str] | None:
    refs: list[str] = []
    for arg in args:
        if command == "mkdir" and arg == "-p":
            continue
        if arg.startswith("-"):
            return None
        refs.append(arg)
    if not refs:
        return set()
    if not all(_is_literal_path(ref) for ref in refs):
        return None
    return set(refs)


def _sed_targets(args: list[str]) -> tuple[set[str] | None, set[str] | None]:
    saw_in_place = False
    consumed_script = False
    refs: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-i":
            saw_in_place = True
            index += 2 if index + 1 < len(args) and args[index + 1] == "" else 1
            continue
        if arg.startswith("-i") and len(arg) > 2:
            saw_in_place = True
            index += 1
            continue
        if arg == "-e":
            if index + 1 >= len(args):
                return None, None
            consumed_script = True
            index += 2
            continue
        if arg.startswith("-"):
            return None, None
        if not consumed_script:
            consumed_script = True
            index += 1
            continue
        refs.append(arg)
        index += 1

    if not saw_in_place or not refs:
        return set(), set()
    if not all(_is_literal_path(ref) for ref in refs):
        return None, None
    return set(refs), set(refs)


def _python_reads(args: list[str]) -> set[str] | None:
    if not args:
        return set()
    if args[:2] == ["-m", "pytest"]:
        return _test_path_reads(args[2:])
    if args[:2] == ["-m", "unittest"]:
        return _test_path_reads(args[2:])

    index = 0
    while index < len(args) and args[index] in {"-u", "-B"}:
        index += 1
    if index >= len(args):
        return set()
    script = args[index]
    if script.startswith("-"):
        return None
    if _is_literal_path(script) and PurePosixPath(script.replace("\\", "/")).suffix == ".py":
        return {script}
    return None


def _node_reads(args: list[str]) -> set[str] | None:
    if not args:
        return set()
    if args[0].startswith("-"):
        return None
    suffix = PurePosixPath(args[0].replace("\\", "/")).suffix
    if suffix in {".js", ".mjs", ".cjs"} and _is_literal_path(args[0]):
        return {args[0]}
    return None


def _npm_reads(args: list[str]) -> set[str] | None:
    if not args:
        return set()
    if args[0] == "test":
        path_args = args[2:] if len(args) >= 2 and args[1] == "--" else []
        return _test_path_reads(path_args)
    if len(args) >= 2 and args[:2] == ["run", "test"]:
        path_args = args[3:] if len(args) >= 3 and args[2] == "--" else []
        return _test_path_reads(path_args)
    return None


def _test_path_reads(args: list[str]) -> set[str] | None:
    refs: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            index += 1
            continue
        if arg in NO_VALUE_FLAGS:
            index += 1
            continue
        if arg in VALUE_FLAGS:
            if index + 1 >= len(args):
                return None
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in VALUE_FLAGS):
            index += 1
            continue
        if arg.startswith("-"):
            return None
        ref = _strip_pytest_nodeid(arg)
        if not _is_literal_path(ref, allow_bare_test_dir=True):
            return None
        refs.append(ref)
        index += 1
    return set(refs)


def _is_literal_path(value: str, *, allow_bare_test_dir: bool = False) -> bool:
    if not value or value in {".", ".."}:
        return False
    if value.startswith("-") or "://" in value:
        return False
    if any(char in value for char in "$`*?[]{}~|;&<>"):
        return False
    normalized = value.replace("\\", "/")
    if normalized.endswith("/..."):
        return False
    if "/" in normalized:
        return True
    if allow_bare_test_dir and normalized in {"test", "tests"}:
        return True
    if PurePosixPath(normalized).suffix or PureWindowsPath(value).suffix:
        return True
    return False


def _strip_pytest_nodeid(value: str) -> str:
    return value.split("::", 1)[0]


def _command_name(value: str) -> str:
    normalized = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if normalized.endswith(".exe"):
        normalized = normalized[:-4]
    return normalized
