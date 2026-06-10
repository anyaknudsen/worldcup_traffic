from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from traffic_client import TrafficAPIClient


EXPECTED_COLUMNS = [
    "timestamp",
    "location_id",
    "congestion_score",
    "travel_time_mins",
    "speed_kph",
    "incident_count",
]


def test_mock_fetch_generates_expected_schema_and_ranges(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    np.random.seed(0)
    client = TrafficAPIClient()

    start_time = datetime(2026, 1, 1, 0, 0)
    end_time = datetime(2026, 1, 1, 5, 0)
    data = client.fetch_traffic_data({"lat": 25.485, "lng": 51.4475}, start_time, end_time)

    assert list(data.columns) == EXPECTED_COLUMNS
    assert len(data) == 6
    assert data["timestamp"].tolist() == list(pd.date_range(start_time, end_time, freq="h"))
    assert set(data["location_id"]) == {"lat_25.485_lng_51.4475"}
    assert data["congestion_score"].between(0, 100).all()
    assert data["speed_kph"].between(20, 60).all()
    assert set(data["incident_count"].unique()).issubset({0, 1})
    assert np.allclose(data["travel_time_mins"], 20 + data["congestion_score"] * 0.8)


def test_fetch_multiple_locations_is_keyed_by_generated_location_id(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    client = TrafficAPIClient()
    start_time = datetime(2026, 1, 1, 0, 0)
    end_time = datetime(2026, 1, 1, 1, 0)

    result = client.fetch_multiple_locations(
        [
            {"city": "Doha", "country": "QA"},
            {"lat": 25.485, "lng": 51.4475},
        ],
        start_time,
        end_time,
    )

    assert set(result) == {"Doha_QA", "lat_25.485_lng_51.4475"}
    assert all(len(frame) == 2 for frame in result.values())


def test_live_api_path_raises_until_implemented():
    client = TrafficAPIClient(api_key="secret")

    with pytest.raises(NotImplementedError, match="Live API not implemented"):
        client.fetch_traffic_data({}, datetime(2026, 1, 1), datetime(2026, 1, 2))
