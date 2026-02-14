

# AGENTS.md – Operational Rules for Codex Agents

## Purpose

This document defines mandatory operational constraints for any AI agent (Codex or similar) modifying this repository.

Agents must follow these rules strictly.

---

## Global Modification Rules

### 1. Do NOT Generate Unit Tests

- Do not create unit tests.
- Do not suggest creating unit tests.
- Do not modify existing files to add test coverage.
- Focus exclusively on the requested implementation changes.

Testing is handled separately and is not part of the agent’s responsibility.

---

### 2. Do NOT Add Fallbacks or Retry Logic

When modifying or creating request logic (HTTP calls, OpenAI calls, third-party APIs, database operations, etc.):

- Do NOT add fallback behavior.
- Do NOT add retry loops.
- Do NOT add silent error handling.
- Do NOT wrap errors to hide failures.

Errors must remain visible.

The system should fail loudly rather than masking issues.

---

## Error Handling Philosophy

- Prefer explicit exceptions.
- Avoid defensive abstractions unless explicitly requested.
- Do not introduce resilience mechanisms without direct instruction.

If a request fails, allow the exception to propagate.

---

## Scope Discipline

When implementing changes:

- Modify only what is explicitly requested.
- Do not refactor unrelated code.
- Do not introduce new architectural patterns.
- Do not add new dependencies.

Keep changes minimal and targeted.

---

## Final Validation Before Completing Changes

Before finishing, ensure:

- No new test files were created.
- No retry logic was introduced.
- No fallback branches were added.
- No silent try/except blocks were added.
- No suppression of exceptions was implemented.

If any of the above were added, remove them.

## Keep Architecture Simple

- Do not introduce abstraction layers unless explicitly requested.
- Do not extract helpers unless code duplication is obvious and large.
- Prefer inline clarity over indirection.
- Prefer readability over cleverness.
- Do not introduce generic try/except blocks.
- Only catch exceptions if explicitly instructed.
- Never use bare except.

## Code Readability

- Prefer explicit variable names.
- Avoid nested conditionals deeper than 2 levels.
- Avoid chained inline expressions.
- Do not use one-liners if they reduce clarity.


## No Silent Defaults

- Do not introduce silent default values.
- Do not auto-correct missing data.
- Do not coerce invalid inputs silently.
- If input is invalid, raise an explicit exception.
- Do not use chained env fallbacks for model selection (for example: `A or B or C`).
- Model configuration must use only `DEFAULT_MODEL` from environment.
- If `DEFAULT_MODEL` is missing, raise an explicit exception.

End of AGENTS.md
