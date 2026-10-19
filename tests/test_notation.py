"""Test T9: Run and verify all doctest examples in docs/notation.md."""

from __future__ import annotations

import doctest
import re
from pathlib import Path
import pytest


def test_t9_notation_doctests():
    """T9: Run every doctest example in docs/notation.md and check against code."""
    docs_file = Path(__file__).parent.parent / "geometry" / "notation.md"
    assert docs_file.exists(), f"Notation file not found at {docs_file}"

    # Extract Python code blocks from markdown, removing ```python and ``` fences
    content = docs_file.read_text(encoding="utf-8")

    # Find all ```python ... ``` blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

    # Join all code blocks with newlines
    doctest_content = "\n\n".join(code_blocks)

    # Execute all doctest blocks
    runner = doctest.DocTestRunner(verbose=False)
    parser = doctest.DocTestParser()
    test = parser.get_doctest(
        doctest_content,
        globs={},
        name="notation.md",
        filename=str(docs_file),
        lineno=0,
    )

    runner.run(test)
    assert runner.tries >= 6, f"Expected at least 6 doctest examples, found {runner.tries}"
    assert runner.failures == 0, f"Doctest failures encountered in {docs_file}: {runner.failures}"
