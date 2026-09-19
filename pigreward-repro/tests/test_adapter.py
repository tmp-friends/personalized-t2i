from pigreward_repro.adapter import parse_judgment, tournament

GOOD = """Composition: Image 1 is balanced. (Score 1: 8, Score 2: 6)
Lighting: Image 2 has softer light. (Score 1: 6, Score 2: 8)
Detail: Image 1 has clearer textures. (Score 1: 9, Score 2: 7)
Score 1: 23
Score 2: 21
Image 1 is better"""


def test_card_format_maps_position_to_actual_image_ids():
    result = parse_judgment(GOOD, ["right", "left"], finished=True)
    assert result["winner_id"] == "right"
    assert result["dimensions"] == ["Composition", "Lighting", "Detail"]
    assert result["status"] == "valid"


def test_invalid_output_never_produces_recommendation():
    for text, finished in [
        (GOOD, False),
        ("tie", True),
        (GOOD.replace("23", "19"), True),
        (GOOD.replace("Image 1 is better", "Image 2 is better"), True),
        ("Image 1 is better", True),
        (GOOD + "\nImage 2 is better", True),
    ]:
        r = parse_judgment(text, ["a", "b"], finished=finished)
        assert r["winner_id"] is None
        assert r["status"] == "inconclusive"


def test_tournament_only_returns_winner_not_ranking():
    calls = []

    def compare(a, b):
        calls.append([a, b])
        return {"status": "valid", "winner_id": a, "reason": "Observed balance."}

    result = tournament(["a", "b", "c", "d"], compare, seed=7)
    assert len(calls) == 3
    assert result["winner_id"] in calls[-1]
    assert "ranking" not in result


def test_tournament_aborts_on_inconclusive_or_foreign_winner():
    for judgment in [
        {"status": "inconclusive", "winner_id": None},
        {"status": "valid", "winner_id": "foreign"},
    ]:
        calls = []

        def compare(a, b, calls=calls, judgment=judgment):
            calls.append([a, b])
            return judgment

        result = tournament(["a", "b", "c", "d"], compare, seed=0)
        assert result["winner_id"] is None
        assert len(calls) == 1
