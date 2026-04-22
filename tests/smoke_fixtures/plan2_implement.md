# Plan 2: String Format Utilities

## Task
Create `strutils/format.py` with the following pure functions:

### `wrap_lines(text: str, width: int) -> str`
Wrap text to the given line width, breaking on spaces.
- Split into words, accumulate into lines that don't exceed width
- Single words longer than width go on their own line (no mid-word breaks)
- Preserve existing newlines as paragraph breaks

Examples:
- `wrap_lines("hello world foo", 10)` → `"hello\nworld foo"` (or similar valid wrapping)
- `wrap_lines("hi", 80)` → `"hi"`
- `wrap_lines("", 10)` → `""`

### `indent(text: str, prefix: str) -> str`
Add a prefix to the start of every line.
- Split on `\n`, prepend prefix to each line, rejoin with `\n`
- Empty lines get the prefix too

Examples:
- `indent("a\nb", "  ")` → `"  a\n  b"`
- `indent("hello", ">>> ")` → `">>> hello"`
- `indent("", "  ")` → `"  "`

## Also Do
1. Create `tests/test_format.py` with at least 5 tests per function using pytest
2. Update `strutils/__init__.py` to also import `wrap_lines` and `indent`
3. Run `python3 -m pytest tests/ -v` to verify ALL tests pass (both plan 1 and plan 2)
