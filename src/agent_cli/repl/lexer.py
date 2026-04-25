"""prompt_toolkit lexer — `!`-prefixed input is rendered primary+bold."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer


class ShellLineLexer(Lexer):
    def lex_document(
        self,
        document: Document,
    ) -> Callable[[int], StyleAndTextTuples]:
        lines = document.lines
        is_shell = document.text.startswith("!")

        def get_line(i: int) -> StyleAndTextTuples:
            text = lines[i] if i < len(lines) else ""
            if is_shell:
                return [("class:shell-line", text)]
            return [("", text)]

        return get_line
