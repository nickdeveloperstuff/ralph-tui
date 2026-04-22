# Plan 1: String Transform Utilities

## Task
Create `strutils/transform.py` with the following pure functions:

### `slugify(text: str) -> str`
Convert a string to a URL-friendly slug.
- Convert to lowercase
- Replace spaces and non-alphanumeric characters with hyphens
- Collapse multiple hyphens into one
- Strip leading/trailing hyphens

Examples:
- `slugify("Hello World")` → `"hello-world"`
- `slugify("  My  Blog Post! ")` → `"my-blog-post"`
- `slugify("foo--bar")` → `"foo-bar"`
- `slugify("")` → `""`

### `truncate(text: str, max_len: int, suffix: str = "...") -> str`
Truncate a string to max_len characters, appending suffix if truncated.
- If `len(text) <= max_len`, return text unchanged
- Otherwise return `text[:max_len - len(suffix)] + suffix`
- If `max_len < len(suffix)`, return `suffix[:max_len]`

Examples:
- `truncate("hello world", 8)` → `"hello..."`
- `truncate("hi", 10)` → `"hi"`
- `truncate("abcdef", 6)` → `"abcdef"`
- `truncate("abcdef", 5)` → `"ab..."`

## Also Do
1. Create `strutils/__init__.py` that imports both functions
2. Create `tests/test_transform.py` with at least 5 tests per function using pytest
3. Run `python3 -m pytest tests/test_transform.py -v` to verify all tests pass
