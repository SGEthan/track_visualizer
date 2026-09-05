from __future__ import annotations

import numpy as np
import pandas as pd

from components.map_view import _dumps
from data_loader import (
    build_trip_summary,
    downsample,
    exclude_airborne_points,
    get_date_range,
    match_flight_windows,
)
from preprocess import prepare_tracks
from preprocess_flights import _parse_ts


def _raw_tracks() -> pd.DataFrame:
    count = 120
    index = np.arange(count)
    timestamps = 1_735_689_600 + index * 20
    timestamps[60:] += 2_000
    return pd.DataFrame({
        "timestamp": timestamps,
        "longitude": -122.4 + index * 0.0002,
        "latitude": 37.7 + np.sin(index / 12) * 0.001,
        "accuracy": np.full(count, 8),
        "stepType": index % 2,
    })


def test_prepare_tracks_accepts_documented_aliases_and_splits_trips() -> None:
    tracks, metadata = prepare_tracks(_raw_tracks())

    assert {"ts", "lon", "lat", "speed", "trip_id", "source"} <= set(tracks.columns)
    assert tracks["ts"].dtype == np.dtype("int64")
    assert metadata["trips"] == 2
    assert tracks["trip_id"].nunique() == 2
    assert (tracks["speed"] <= 300).all()


def test_downsample_never_exceeds_the_requested_cap() -> None:
    tracks, _ = prepare_tracks(_raw_tracks())
    many = pd.concat(
        [tracks.assign(trip_id=trip_id, ts=tracks["ts"] + trip_id * 10_000)
         for trip_id in range(30)],
        ignore_index=True,
    )

    sampled = downsample(many, 17)

    assert len(sampled) == 17


def test_trip_summary_is_non_negative_and_complete() -> None:
    tracks, _ = prepare_tracks(_raw_tracks())

    summary = build_trip_summary(tracks)

    assert len(summary) == 2
    assert int(summary["point_count"].sum()) == len(tracks)
    assert (summary["distance_km"] >= 0).all()
    assert (summary["duration_min"] >= 0).all()


def test_embedded_json_cannot_close_the_script_tag() -> None:
    serialized = _dumps({"value": "</script><script>alert(1)</script>"})

    assert "</script>" not in serialized
    assert "\\u003c/script" in serialized


def test_date_range_includes_imported_flights() -> None:
    tracks, _ = prepare_tracks(_raw_tracks())

    low, high = get_date_range(
        tracks,
        [{"date": "2020-01-01"}, {"date": "2026-08-24"}, {"date": "invalid"}],
    )

    assert low.isoformat() == "2020-01-01"
    assert high.isoformat() == "2026-08-24"


def test_flighty_local_time_uses_airport_timezone() -> None:
    utc = _parse_ts("2025-01-01T10:00", "UTC")
    los_angeles = _parse_ts("2025-01-01T10:00", "America/Los_Angeles")

    assert utc is not None and los_angeles is not None
    assert los_angeles - utc == 8 * 3600


def test_match_flight_windows_returns_review_candidates_without_mutation() -> None:
    tracks, _ = prepare_tracks(_raw_tracks())
    original_columns = tracks.columns.tolist()
    departure = int(tracks["ts"].iloc[10])
    arrival = int(tracks["ts"].iloc[20])

    candidates = match_flight_windows(tracks, [{
        "id": "test-flight",
        "date": "2025-01-01",
        "from_iata": "SFO",
        "to_iata": "SEA",
        "airline": "AS",
        "flight": "1",
        "dep_ts": departure,
        "arr_ts": arrival,
        "dep_time_source": "Take off (Actual)",
        "arr_time_source": "Landing (Actual)",
        "canceled": False,
    }])

    assert len(candidates) == 11
    assert candidates["flight_match_id"].eq("test-flight").all()
    assert "flight_match_id" not in tracks.columns
    assert tracks.columns.tolist() == original_columns


def test_exclude_airborne_points_removes_and_resegments_without_mutation() -> None:
    tracks, _ = prepare_tracks(_raw_tracks())
    tracks = tracks.assign(trip_id=np.int32(0))
    candidates = tracks.iloc[55:66].copy()

    ground = exclude_airborne_points(tracks, candidates)

    assert len(ground) == len(tracks) - len(candidates)
    assert ground.index.intersection(candidates.index).empty
    assert ground["trip_id"].nunique() >= 2
    assert tracks["trip_id"].eq(0).all()
