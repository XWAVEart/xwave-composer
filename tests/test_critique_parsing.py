"""A 3B model ignores "JSON only" often enough that the parser is the risk."""

from xwave_composer.models.llm_rewriter import _parse_critique_json


def test_plain_json():
    got = _parse_critique_json('{"critique": "too dark", "improved_prompt": "bright sunlit"}')
    assert got == ("too dark", "bright sunlit")


def test_fenced_json():
    raw = '```json\n{"critique": "flat light", "improved_prompt": "rim lit"}\n```'
    assert _parse_critique_json(raw) == ("flat light", "rim lit")


def test_unlabelled_fence():
    raw = '```\n{"critique": "a", "improved_prompt": "b"}\n```'
    assert _parse_critique_json(raw) == ("a", "b")


def test_json_embedded_in_prose():
    raw = 'Sure! Here is my analysis:\n{"critique": "muddy", "improved_prompt": "crisp"}\nHope that helps.'
    assert _parse_critique_json(raw) == ("muddy", "crisp")


def test_camelcase_key_from_the_original_workflow():
    raw = '{"critique": "x", "improvedPrompt": "y"}'
    assert _parse_critique_json(raw) == ("x", "y")


def test_critique_returned_as_a_list_of_bullets():
    raw = '{"critique": ["no boat", "wrong time of day"], "improved_prompt": "z"}'
    critique, improved = _parse_critique_json(raw)
    assert "no boat" in critique and "wrong time of day" in critique
    assert improved == "z"


def test_multiline_critique_survives():
    raw = '{"critique": "- one\\n- two", "improved_prompt": "p"}'
    critique, _ = _parse_critique_json(raw)
    assert critique.count("\n") == 1


def test_prose_only_is_rejected():
    assert _parse_critique_json("The image looks pretty good to me, honestly.") is None


def test_missing_improved_prompt_is_rejected():
    assert _parse_critique_json('{"critique": "only half of it"}') is None


def test_empty_is_rejected():
    assert _parse_critique_json("") is None


def test_non_object_json_is_rejected():
    assert _parse_critique_json('["not", "an", "object"]') is None
