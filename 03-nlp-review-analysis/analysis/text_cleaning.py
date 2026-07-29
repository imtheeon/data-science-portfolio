"""Shared text-cleaning step applied once, before both sentiment scoring and
topic modeling, so review text is cleaned consistently and only in one
place.

Amazon review text scraped from the source dataset contains literal HTML
markup (most commonly stray `<br />` line breaks). Left in place, these tags
get vectorized as tokens like "br" alongside real words, which can dominate
a topic model (e.g. produce a whole "topic" that is just the token "br")
and slightly skew sentiment scoring."""

from __future__ import annotations

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags from ``text``, replacing them with a space so words
    on either side of a tag don't get glued together, then collapse the
    resulting run(s) of whitespace. Non-string input is returned unchanged."""
    if not isinstance(text, str):
        return text
    cleaned = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()
