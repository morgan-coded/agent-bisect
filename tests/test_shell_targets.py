from agent_bisect.shell_targets import ShellTargets, extract_shell_targets


def test_extract_shell_targets_whitelist():
    cases = [
        ("cat repo/input.txt", ["repo/input.txt"], []),
        ("cat repo/a::b.py", ["repo/a::b.py"], []),
        ("less repo/input.txt", ["repo/input.txt"], []),
        ("head -n 5 repo/input.txt", ["repo/input.txt"], []),
        ("tail -20 repo/input.txt", ["repo/input.txt"], []),
        ("python repo/build.py > repo/out.txt", ["repo/build.py"], ["repo/out.txt"]),
        ("node repo/app.js", ["repo/app.js"], []),
        ("pytest repo/test_app.py::test_demo", ["repo/test_app.py"], []),
        ("python -m pytest tests", ["tests"], []),
        ("go test ./pkg", ["./pkg"], []),
        ("npm test -- repo/ui.test.js", ["repo/ui.test.js"], []),
        ("tee repo/out.txt", [], ["repo/out.txt"]),
        ("cp repo/a.txt repo/b.txt", ["repo/a.txt"], ["repo/b.txt"]),
        ("mv repo/a.txt repo/b.txt", ["repo/a.txt"], ["repo/a.txt", "repo/b.txt"]),
        ("rm -f repo/a.txt", [], ["repo/a.txt"]),
        ("touch repo/new.txt", [], ["repo/new.txt"]),
        ("mkdir -p repo/cache", [], ["repo/cache"]),
        ("sed -i 's/old/new/' repo/app.py", ["repo/app.py"], ["repo/app.py"]),
        ("sed -i '' 's/old/new/' repo/app.py", ["repo/app.py"], ["repo/app.py"]),
    ]

    for command, reads, writes in cases:
        assert extract_shell_targets(command) == ShellTargets(reads=reads, writes=writes)


def test_extract_shell_targets_ambiguous_cases_are_empty():
    cases = [
        "cat repo/input.txt | grep x",
        "cat $(pwd)/input.txt",
        "cat `pwd`/input.txt",
        "cat $LOG > repo/out.txt",
        "cat repo/*.txt",
        "cat repo/input?.txt",
        "cat repo/[ab].txt",
        "cat repo/{a,b}.txt",
        "cat ~/repo/input.txt",
        "cat repo/a.txt && cat repo/b.txt",
        "cat repo/a.txt & cat repo/b.txt",
        "cat repo/a.txt; cat repo/b.txt",
        "cat repo/a.txt\ncat repo/b.txt",
        "cat --unknown repo/a.txt",
        "pytest --unknown repo/test_app.py",
        "cp -r repo/a repo/b",
        "sed -r 's/old/new/' repo/app.py",
        "sed 's/old/new/' repo/app.py",
        None,
        7,
        "",
    ]

    for command in cases:
        assert extract_shell_targets(command) == ShellTargets(reads=[], writes=[])


def test_extract_shell_targets_is_deterministic():
    command = "cp repo/b.txt repo/a.txt"

    first = extract_shell_targets(command)
    second = extract_shell_targets(command)

    assert first == second
