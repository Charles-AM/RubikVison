from __future__ import annotations

from app.app import parse_terminal_key


def test_parse_terminal_key_accepts_color_command() -> None:
    assert parse_terminal_key(" W\n") == ord("w")


def test_parse_terminal_key_uses_first_character() -> None:
    assert parse_terminal_key("red\n") == ord("r")


def test_parse_terminal_key_ignores_empty_command() -> None:
    assert parse_terminal_key("  \n") is None

