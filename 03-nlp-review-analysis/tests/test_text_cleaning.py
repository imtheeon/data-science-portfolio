from analysis.text_cleaning import strip_html


def test_strip_html_removes_br_tags():
    assert strip_html("Great taste!<br />Would buy again.") == "Great taste! Would buy again."


def test_strip_html_removes_arbitrary_tags():
    assert strip_html("<b>Bold</b> and <i>italic</i> text") == "Bold and italic text"


def test_strip_html_collapses_whitespace_left_by_tags():
    assert strip_html("word1<br/><br/>word2") == "word1 word2"


def test_strip_html_no_tags_is_unchanged():
    assert strip_html("Plain text, no markup here.") == "Plain text, no markup here."


def test_strip_html_non_string_passthrough():
    assert strip_html(None) is None
