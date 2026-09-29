"""Exact-match offline evaluation helper."""
def exact_match(actual: str, expected: str) -> bool:
    return actual.strip() == expected.strip()
