"""Tests for data/odds.py: the canonical odds schema and its join.

The cards over/under line comes from the odds source, so this boundary decides
what the rest of the system can price. Two properties matter and are pinned here:
the schema is enforced (a half-usable file raises rather than pricing some rows
off garbage), and the join REPORTS every unmatched odds row and every unpriceable
match instead of dropping either silently — a silent drop would bias every ROI
that follows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bet_engine.data.odds import (
    ODDS_COLUMNS,
    apply_team_names,
    join_odds,
    load_odds_table,
    normalise_odds,
    select_odds,
)


def _raw(**overrides):
    row = {
        "date": "2021-08-14",
        "home": "Manchester United",
        "away": "Leeds United",
        "line": 4.5,
        "over_odds": 1.9,
        "under_odds": 1.95,
        "source": "pinnacle",
        "is_closing": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# --- normalise_odds ---------------------------------------------------------


def test_normalise_returns_the_canonical_schema_in_order():
    out = normalise_odds(_raw())

    assert list(out.columns) == list(ODDS_COLUMNS)
    assert str(out["date"].dtype).startswith("datetime64")
    assert out["is_closing"].dtype == bool
    assert out.loc[0, "is_closing"] == True  # noqa: E712 - a numpy bool, not identity
    assert out.loc[0, "line"] == 4.5


def test_normalise_defaults_the_optional_columns():
    out = normalise_odds(_raw().drop(columns=["source", "is_closing"]))

    assert bool(out.loc[0, "is_closing"]) is False
    assert pd.isna(out.loc[0, "source"])
    assert pd.isna(out.loc[0, "match_id"])  # our matches own the id


def test_normalise_strips_text_and_reads_textual_booleans():
    out = normalise_odds(_raw(home="  Man City  ", is_closing="closing"))

    assert out.loc[0, "home"] == "Man City"
    assert bool(out.loc[0, "is_closing"]) is True


@pytest.mark.parametrize(
    "frame, message",
    [
        (_raw().drop(columns=["line"]), "missing required column"),
        (_raw(date="not-a-date"), "unparseable date"),
        (_raw(home="   "), "blank home"),
        (_raw(away=None), "blank away"),
        (_raw(line=-1.0), "line must be finite"),
        (_raw(line=float("inf")), "line must be finite"),
        (_raw(over_odds=1.0), "over_odds must be a finite price"),
        (_raw(under_odds=float("nan")), "under_odds must be a finite price"),
    ],
)
def test_normalise_refuses_a_half_usable_table(frame, message):
    with pytest.raises(ValueError, match=message):
        normalise_odds(frame)


def test_normalise_rejects_non_frames():
    with pytest.raises(TypeError):
        normalise_odds([{"date": "2021-08-14"}])


# --- team-name mapping ------------------------------------------------------


def _write_map(tmp_path, aliases):
    path = tmp_path / "team_names.yaml"
    lines = ["aliases:"]
    for source, target in aliases.items():
        lines.append(f'  "{source}": "{target}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_apply_team_names_renames_by_the_mapping_file(tmp_path):
    path = _write_map(tmp_path, {"Manchester United": "Man United"})
    frame = normalise_odds(_raw())

    out = apply_team_names(frame, path)

    assert out.loc[0, "home"] == "Man United"
    assert out.loc[0, "away"] == "Leeds United"  # unmapped names pass through
    # The input frame is not mutated: raw data is never edited.
    assert frame.loc[0, "home"] == "Manchester United"


def test_a_missing_mapping_file_is_an_error_not_a_silent_no_op():
    with pytest.raises(FileNotFoundError, match="team-name map not found"):
        apply_team_names(normalise_odds(_raw()), "does/not/exist.yaml")


# --- load_odds_table --------------------------------------------------------


def test_load_reads_normalises_and_maps(tmp_path):
    csv = tmp_path / "odds.csv"
    _raw().to_csv(csv, index=False)
    mapping = _write_map(tmp_path, {"Manchester United": "Man United"})

    out = load_odds_table(csv, team_names=mapping)

    assert list(out.columns) == list(ODDS_COLUMNS)
    assert out.loc[0, "home"] == "Man United"


# --- select_odds ------------------------------------------------------------


def test_select_prefers_the_closing_price_per_match():
    odds = normalise_odds(
        pd.concat(
            [
                _raw(over_odds=2.0, is_closing=False),
                _raw(over_odds=1.9, is_closing=True),
            ]
        )
    )

    chosen = select_odds(odds)

    assert len(chosen) == 1
    assert chosen.loc[0, "over_odds"] == 1.9
    assert bool(chosen.loc[0, "is_closing"]) is True


def test_select_without_the_closing_preference_keeps_the_first_row():
    odds = normalise_odds(
        pd.concat([_raw(over_odds=2.0), _raw(over_odds=1.9)])
    )

    chosen = select_odds(odds, prefer_closing=False)

    assert chosen.loc[0, "over_odds"] == 2.0


# --- join_odds --------------------------------------------------------------


def _matches():
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": pd.to_datetime(["2021-08-14", "2021-08-15"]),
            "home": ["Man United", "Man City"],
            "away": ["Leeds United", "Tottenham"],
        }
    )


def test_join_attaches_the_odds_columns():
    odds = normalise_odds(
        _raw(home="Man United", away="Leeds United", match_id="o1")
    )

    merged, report = join_odds(_matches(), odds)

    assert report.n_odds == 1
    assert report.n_matches == 2
    assert report.n_matched == 1
    row = merged.loc[merged["match_id"] == "m1"].iloc[0]
    assert row["line"] == 4.5
    assert row["over_odds"] == 1.9
    assert row["odds_match_id"] == "o1"
    # m2 has no odds: prices and line are NaN, reported as an unmatched match.
    m2 = merged.loc[merged["match_id"] == "m2"].iloc[0]
    assert pd.isna(m2["line"])
    assert m2["match_id"] == "m2"  # our id is untouched by the join


def test_join_reports_every_unmatched_odds_row_and_every_unpriced_match():
    odds = normalise_odds(
        pd.concat(
            [
                _raw(home="Man United", away="Leeds United"),
                _raw(home="Ghost FC", away="Nowhere United"),
            ]
        )
    )

    merged, report = join_odds(_matches(), odds)

    assert report.n_unmatched_odds == 1
    assert report.unmatched_odds["home"].tolist() == ["Ghost FC"]
    # Both of our matches lack odds here (the odds only cover m1's pair).
    assert report.n_unmatched_matches == 1
    assert report.unmatched_matches["match_id"].tolist() == ["m2"]
    assert len(merged) == 2


def test_join_requires_our_matches_to_carry_the_join_keys():
    matches = _matches().drop(columns="home")
    with pytest.raises(ValueError, match="matches is missing join column"):
        join_odds(matches, normalise_odds(_raw()))


def test_join_requires_a_normalised_odds_frame():
    raw = _raw()  # has no `match_id`/canonical columns yet
    with pytest.raises(ValueError, match="canonical column"):
        join_odds(_matches(), raw)
