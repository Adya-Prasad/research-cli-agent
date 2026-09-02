# Memory Recall CLI Crash — Bugfix Design

## Overview

The `memory recall` CLI command crashes at runtime with `ValueError: Format specifier missing
precision` whenever it attempts to render search results in a Rich table. A stray space in
the f-string format specifier (`":. 4f"` instead of `":.4f"`) on the score column is the
primary defect. The same file also contains a mismatched Rich markup tag in the `forget`
success message (`[bold gree]` vs `[bold green]`) and a duplicate `MemoryService` import.

The fix is surgical: correct three isolated code-quality issues in
`src/research_agent/memory/cli.py` without altering any public behaviour, CLI interface,
or data model.

## Glossary

- **Bug_Condition (C)**: The set of runtime inputs that trigger the crash — specifically,
  any invocation of `memory recall` that receives at least one memory hit from the service,
  causing the malformed f-string to be evaluated.
- **Property (P)**: The desired correct behaviour for those inputs — the Rich table is
  rendered without error, scores are formatted to four decimal places, and the command
  exits with code 0.
- **Preservation**: All CLI interactions that do NOT involve rendering the recall results
  table must produce exactly the same observable output and exit codes as before the fix.
- **`recall`**: The Typer command in `cli.py` that calls `MemoryService.recall` and renders
  results in a Rich table.
- **`forget`**: The Typer command in `cli.py` that deletes a memory and prints a success
  message using Rich markup.
- **`hit.score`**: A `float` field on each `MemoryHit` result, formatted for display in the
  "Score" column of the recall table.

## Bug Details

### Bug Condition

The crash manifests when `memory recall` is invoked, the service returns one or more hits,
and Python evaluates the f-string `f"{hit.score:. 4f}"`. The stray space between `.` and
`4` is an invalid format specification, so Python raises `ValueError: Format specifier
missing precision` before any row is added to the table.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input — a CLI invocation context
  OUTPUT: boolean

  RETURN input.command == "recall"
         AND MemoryService.recall(input.user_id, input.query, input.top_k) returns len >= 1
         AND score_format_specifier contains stray space (":. 4f")
END FUNCTION
```

### Examples

- **Primary crash**: `memory recall --user-id user-1 "agent memory systems"` — service
  returns one hit with `score=0.9231`; Python raises `ValueError` when evaluating
  `f"{0.9231:. 4f}"`. Expected: `"0.9231"` rendered in the Score column; exit code 0.
- **Multi-hit crash**: Same command returning three hits — crashes on the first row.
  Expected: all rows rendered, table printed, exit code 0.
- **Typo in forget message**: `memory forget --user-id user-1 <id>` succeeds but renders
  `[bold gree]Memory deleted[/bold green]` — mismatched tag. Expected: `[bold green]`.
- **Duplicate import**: Importing `cli.py` evaluates `from research_agent.memory.service
  import MemoryService` twice. Expected: exactly one import statement.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `memory recall` with zero hits MUST continue to print the "No memories found." warning
  and exit with code 0.
- `memory remember` with valid, policy-approved content MUST continue to persist the memory
  and print "Memory accepted" with the assigned ID and provider status, exiting code 0.
- `memory remember` with policy-violating content MUST continue to print the rejection
  reason and exit with code 1.
- `memory forget` for an unknown or mismatched memory ID MUST continue to print
  "Memory not found or ownership mismatch." and exit with code 1.
- `memory recall` with a custom `--top-k` value MUST continue to pass that value to the
  underlying service and return at most that many results.

**Scope:**
All inputs that do NOT involve rendering at least one recall result row are completely
unaffected by this fix. This includes:

- All `memory remember` invocations (accepted or rejected).
- `memory recall` returning zero results.
- `memory forget` invocations (success or failure paths).
- Any other CLI sub-commands outside the `memory` group.

## Hypothesized Root Cause

1. **Malformed f-string format specifier** (`":. 4f"`): A stray space was introduced
   between `.` and `4` in the precision specifier. Python's `str.format_map` / f-string
   evaluation rejects this at runtime, raising `ValueError`. This is the sole cause of the
   test failure (`recalled.exit_code == 1`).

2. **Mismatched Rich markup tag** (`[bold gree]`): The opening tag in the `forget` success
   `console.print` call is `gree` instead of `green`. Rich may silently drop unrecognised
   tags, so the text may still appear — but the markup is semantically wrong and will
   produce unexpected rendering in terminals with strict Rich versions.

3. **Duplicate import statement**: Lines 16–17 both import `MemoryService` from the same
   module. While Python deduplicates module imports internally, the redundant line is dead
   code that creates confusion and triggers linter warnings.

## Correctness Properties

Property 1: Bug Condition — Recall Table Renders Without Error

_For any_ CLI invocation where `isBugCondition` returns true (i.e., `memory recall` is
called and the service returns at least one hit), the fixed `recall` command SHALL format
each hit's score with `:.4f`, render all rows in the Rich table without raising any
exception, print the table to stdout, and exit with code 0.

**Validates: Requirements 2.1**

Property 2: Preservation — Non-Buggy Invocations Are Unchanged

_For any_ CLI invocation where `isBugCondition` returns false (zero recall results, any
`remember` call, any `forget` call, other commands), the fixed code SHALL produce
exactly the same stdout output, stderr output, and exit code as the original code,
preserving all existing functionality for non-recall-result-rendering interactions.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

**File**: `src/research_agent/memory/cli.py`

**Specific Changes:**

1. **Remove duplicate import (lines 16–17)**
   - Delete the second occurrence of `from research_agent.memory.service import MemoryService`.
   - Keep only one import statement.

2. **Fix score format specifier in `recall` (line ~116)**
   - Change `f"{hit.score:. 4f}"` → `f"{hit.score:.4f}"`
   - Remove the stray space between `.` and `4` so Python can parse the precision correctly.

3. **Fix mismatched Rich markup tag in `forget` (line ~143)**
   - Change `"[bold gree]Memory deleted[/bold green]"` → `"[bold green]Memory deleted[/bold green]"`
   - Align the opening tag with the closing tag so Rich renders bold green correctly.

No other files need to be changed. No public interfaces, data models, or CLI signatures
are altered.

## Testing Strategy

### Validation Approach

The testing strategy follows two phases: first confirm the bug exists by running the
current failing test against unfixed code, then apply the fix and verify all tests pass
and no regressions are introduced.

### Exploratory Bug Condition Checking

**Goal**: Surface the counterexample that demonstrates the crash BEFORE implementing the
fix — confirming the root cause is the format specifier and not something deeper (e.g., a
service-layer issue or a Rich version incompatibility).

**Test Plan**: Run the existing integration test on unfixed code and observe the
`ValueError` traceback. Optionally write a focused unit-level test that directly evaluates
`f"{0.9231:. 4f}"` to confirm it raises `ValueError`.

**Test Cases:**
1. **Format specifier unit test**: Evaluate `f"{0.9231:. 4f}"` and assert `ValueError` is
   raised — confirms Python rejects the specifier (will fail/pass as expected on unfixed
   code depending on framing).
2. **Recall integration test**: Invoke `memory recall` with a monkeypatched service
   returning one hit and assert `exit_code == 1` and output contains no table — confirms
   the bug is observable at the CLI level on unfixed code.

**Expected Counterexamples:**
- `ValueError: Format specifier missing precision` raised during table row construction.
- Possible root causes: invalid format string, not a service error, not a Rich version issue.

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed command
produces the expected output and exit code 0.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := recall_fixed(input)
  ASSERT result.exit_code == 0
  ASSERT score formatted as "X.XXXX" appears in result.output
  ASSERT result.output contains the memory content
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed
command produces exactly the same observable behaviour as before.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT recall_original(input) output == recall_fixed(input) output
  ASSERT recall_original(input) exit_code == recall_fixed(input) exit_code
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking
because:
- It generates many random inputs across the full CLI parameter space.
- It catches unexpected regressions in edge cases (empty queries, max `top_k`, special
  characters in content, etc.).
- It provides strong guarantees that non-buggy paths are untouched.

**Test Plan**: Run the existing `test_memory_cli_reports_policy_rejection` test (covers
`remember` rejection) and the zero-results path of recall; then write property-based tests
for the `remember` and `forget` happy paths with randomised inputs.

**Test Cases:**
1. **Zero-results recall preservation**: Invoke `memory recall` with a service returning
   `[]` and verify output contains "No memories found." and exit code 0 — unchanged.
2. **Remember acceptance preservation**: Invoke `memory remember` with random
   policy-approved content and verify exit code 0 and "Memory accepted" in output.
3. **Remember rejection preservation**: Invoke `memory remember` with policy-violating
   content and verify exit code 1 and rejection reason in output.
4. **Forget not-found preservation**: Invoke `memory forget` with an unknown ID and verify
   exit code 1 and "Memory not found or ownership mismatch." in output.
5. **Forget success message**: Invoke `memory forget` for an existing ID and verify output
   contains "Memory deleted" rendered correctly (validates the markup tag fix too).

### Unit Tests

- Test that `f"{score:.4f}"` formats correctly for representative float values
  (0.0, 1.0, 0.5, 0.9999, very small values).
- Test that the `forget` success path prints a message containing "Memory deleted".
- Test that the `cli.py` module has no duplicate top-level import statements (static
  analysis or AST inspection).

### Property-Based Tests

- Generate random lists of `MemoryHit` objects with arbitrary float scores and verify the
  recall table renders without error and all scores appear formatted to four decimal places.
- Generate random valid memory content strings and verify `memory remember` always exits
  with code 0 when policy approves them, regardless of content phrasing.
- Generate random memory IDs and user IDs and verify `memory forget` for non-existent IDs
  always exits with code 1 without crashing.

### Integration Tests

- Full round-trip: `remember` → `recall` (assert exit 0 and content in output) → `forget`
  → `recall` (assert "No memories found." and exit 0). This is the existing
  `test_memory_cli_supports_remember_recall_and_forget` test, which must pass after the fix.
- Verify `--top-k` parameter is respected: remember 5 memories, recall with `--top-k 2`,
  assert at most 2 rows appear in the output.
- Verify policy rejection continues to work alongside a successful remember in the same
  service instance.
