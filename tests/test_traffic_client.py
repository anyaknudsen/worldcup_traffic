from datetime import datetime
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

import traffic_client
from traffic_client import DEFAULT_TRAFFIC_API_KEY, TrafficAPIClient


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
    client = TrafficAPIClient(api_key="")

    start_time = datetime(2026, 1, 1, 0, 0)
    end_time = datetime(2026, 1, 1, 5, 0)
    data = client.fetch_traffic_data({"lat": 25.485, "lng": 51.4475}, start_time, end_time)

    assert list(data.columns) == EXPECTED_COLUMNS
    assert len(data) == 6
    assert data["timestamp"].tolist() == list(pd.date_range(start_time, end_time, freq="h"))
    assert set(data["location_id"]) == {"lat_25.485_lng_51.4475"}
    assert data["congestion_score"].between(0, 100).all()
    assert data["travel_time_mins"].between(20, 100).all()
    assert data["speed_kph"].between(20, 60).all()
    assert set(data["incident_count"].unique()).issubset({0, 1})


def test_mock_speed_and_travel_time_move_plausibly_with_congestion(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    np.random.seed(123)
    client = TrafficAPIClient(api_key="")

    start_time = datetime(2026, 1, 1, 0, 0)
    end_time = datetime(2026, 1, 14, 23, 0)
    data = client.fetch_traffic_data({"city": "Doha", "country": "QA"}, start_time, end_time)

    ordered = data.sort_values("congestion_score")
    assert ordered["travel_time_mins"].is_monotonic_increasing
    assert ordered["speed_kph"].is_monotonic_decreasing
    assert data["congestion_score"].corr(data["travel_time_mins"]) > 0.9
    assert data["congestion_score"].corr(data["speed_kph"]) < -0.9


def test_fetch_multiple_locations_is_keyed_by_generated_location_id(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    client = TrafficAPIClient(api_key="")
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


def test_default_constructor_uses_configured_tomtom_key(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)

    client = TrafficAPIClient()

    assert client.api_key == DEFAULT_TRAFFIC_API_KEY
    assert client.use_mock is False


def test_live_fetch_calls_tomtom_flow_and_incidents(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return self.payload.encode("utf-8")

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        if "flowSegmentData" in url:
            return FakeResponse(
                """
                {
                    "flowSegmentData": {
                        "currentSpeed": 40,
                        "freeFlowSpeed": 80,
                        "currentTravelTime": 120
                    }
                }
                """
            )
        if "incidentDetails" in url:
            return FakeResponse(
                """
                {
                    "incidents": [
                        {"properties": {"id": "one"}},
                        {"properties": {"id": "two"}}
                    ]
                }
                """
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(traffic_client, "urlopen", fake_urlopen)
    client = TrafficAPIClient(api_key="test-key", request_timeout=5)

    data = client.fetch_traffic_data(
        {"lat": 25.485, "lng": 51.4475},
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 2, 0),
    )

    assert list(data.columns) == EXPECTED_COLUMNS
    assert len(data) == 3
    assert data["speed_kph"].tolist() == [40, 40, 40]
    assert data["travel_time_mins"].tolist() == [2.0, 2.0, 2.0]
    assert data["incident_count"].tolist() == [2, 2, 2]
    assert data["congestion_score"].tolist() == [50.0, 50.0, 50.0]

    assert len(calls) == 2
    assert all(timeout == 5 for _, timeout in calls)
    flow_url, incident_url = [url for url, _ in calls]
    flow_query = parse_qs(urlparse(flow_url).query)
    incident_query = parse_qs(urlparse(incident_url).query)
    assert flow_query["point"] == ["25.485,51.4475"]
    assert flow_query["unit"] == ["KMPH"]
    assert incident_query["timeValidityFilter"] == ["present"]
    assert incident_query["bbox"] == ["51.4375,25.475,51.4575,25.495"]


def test_live_fetch_requires_coordinates():
    client = TrafficAPIClient(api_key="secret")

    with pytest.raises(ValueError, match="requires lat and lng"):
        client.fetch_traffic_data(
            {"city": "Doha", "country": "QA"},
            datetime(2026, 1, 1),
            datetime(2026, 1, 2),
        )
