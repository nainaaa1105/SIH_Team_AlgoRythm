"""The login/signup form's mobile layout.

Bug report: the dashboard "only worked properly on laptop." Root cause,
found by inspection: `.lp-login` (the box both the login and signup
steps use) has a fixed `width: 420px` plus a `margin-left: 12vw`
desktop-only offset (to sit beside the intro globe animation). On a
typical phone (375-414px wide) that's wider than the viewport itself --
the form overflowed off-screen rather than just being cramped, and since
it's the very first screen, a broken login form breaks the whole mobile
experience regardless of how well anything past it responds.
"""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def _phone_breakpoint_body(src: str) -> str:
    """The @media (max-width: 640px) block's own body -- other phone
    fixes already live here (#dp width, #bb wrapping), so the login-form
    fix belongs alongside them rather than in a new breakpoint."""
    start = src.index("@media (max-width: 640px)")
    # Balance braces from the block's own opening one to find its close.
    depth = 0
    i = src.index("{", start)
    block_start = i
    for i in range(i, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[block_start:i]
    raise AssertionError("unbalanced @media (max-width: 640px) block")


def test_login_form_shrinks_to_fit_a_phone_viewport():
    block = _phone_breakpoint_body(html())
    assert ".lp-login" in block
    lp_login_rule = block.split(".lp-login {")[1].split("}")[0]
    assert "min(420px, 88vw)" in lp_login_rule


def test_login_form_desktop_offset_is_cleared_not_just_shrunk():
    """The desktop 12vw margin-left exists to sit the box beside the
    globe visual -- on a screen too narrow for both, that offset alone
    (even with a smaller width) still pushes the box off-center or
    off-screen. Must be zeroed and the box centered instead."""
    block = _phone_breakpoint_body(html())
    lp_login_rule = block.split(".lp-login {")[1].split("}")[0]
    assert "margin-left: auto" in lp_login_rule
    assert "margin-right: auto" in lp_login_rule


def test_signup_reuses_the_same_fixed_class():
    """Confirms the one CSS fix covers both screens -- signup has no
    separate class of its own that would need its own fix."""
    src = html()
    assert 'class="lp-login" id="lp-role-login"' in src
    assert 'class="lp-login" id="lp-role-signup"' in src


def test_tagline_is_cut_down_to_fit_instead_of_overflowing():
    """Uppercase + 5px letter-spacing on a ~35-character line is wider
    than most phone screens regardless of font-size alone."""
    block = _phone_breakpoint_body(html())
    assert ".th-tagline" in block
    tagline_rule = block.split(".th-tagline {")[1].split("}")[0]
    assert "letter-spacing" in tagline_rule
    px_match = re.search(r"font-size:\s*(\d+)px", tagline_rule)
    assert px_match and int(px_match.group(1)) < 14  # smaller than the desktop 14px
