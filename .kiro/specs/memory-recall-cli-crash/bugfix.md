# Bugfix Requirements Document

## Introduction

The `memory recall` CLI command crashes at runtime with `ValueError: Format specifier missing precision` whenever it attempts to render search results in a Rich table. The crash is caused by a stray space in an f-string format specifier (`":. 4f"` instead of `":.4f"`). Because the exception propagates unhandled through Typer, the command exits with code 1 even when valid memories exist — making recall completely non-functional.

Two additional defects exist in the same file: a misspelled Rich markup tag in the `forget` success message (`[bold gree]` instead of `[bold green]`), and a duplicate import statement for `MemoryService`.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `memory recall` is invoked and at least one memory hit is returned THEN the system raises `ValueError: Format specifier missing precision` while formatting the score column with `f"{hit.score:. 4f}"` (stray space between `.` and `4`), causing Typer to exit with code 1.

1.2 WHEN `memory forget` succeeds in deleting a memory THEN the system prints `[bold gree]Memory deleted[/bold green]` with a mismatched opening tag (`gree` instead of `green`), producing malformed Rich markup in the success message.

1.3 WHEN the `cli.py` module is imported THEN `from research_agent.memory.service import MemoryService` is evaluated twice due to a duplicate import on consecutive lines, producing unnecessary module-level redundancy.

### Expected Behavior (Correct)

2.1 WHEN `memory recall` is invoked and at least one memory hit is returned THEN the system SHALL render a Rich table with scores formatted to four decimal places using the correct specifier `:.4f` and exit with code 0.

2.2 WHEN `memory forget` succeeds in deleting a memory THEN the system SHALL print `[bold green]Memory deleted[/bold green]` with a correctly matched opening tag so Rich renders the text in bold green.

2.3 WHEN the `cli.py` module is imported THEN `MemoryService` SHALL be imported exactly once, with the duplicate import statement removed.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `memory recall` is invoked and no memories are found THEN the system SHALL CONTINUE TO print the "No memories found." warning and exit with code 0.

3.2 WHEN `memory remember` is invoked with valid, policy-approved content THEN the system SHALL CONTINUE TO persist the memory and print the "Memory accepted" confirmation with the assigned ID and provider status.

3.3 WHEN `memory remember` is invoked with content that violates policy THEN the system SHALL CONTINUE TO print the rejection reason and exit with code 1.

3.4 WHEN `memory forget` is invoked for a memory ID that does not exist or belongs to a different user THEN the system SHALL CONTINUE TO print the "Memory not found or ownership mismatch." error and exit with code 1.

3.5 WHEN `memory recall` is invoked with a custom `--top-k` value THEN the system SHALL CONTINUE TO pass that value to the underlying service and return at most that many results.
