"""
Traffic API client for fetching traffic data.

When no live API key is available, the client generates mock traffic data for
demonstration and tests.
"""

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TRAFFIC_API_KEY = "jqEBfegxUXu6zwI1jdY8J03mZjxrDgV5"
TOMTOM_TRAFFIC_BASE_URL = "https://api.tomtom.com/traffic/services"
TOMTOM_INCIDENT_FIELDS = (
    "{incidents{type,geometry{type,coordinates},properties{id,iconCategory}}}"
)
DEFAULT_INCIDENT_RADIUS_DEGREES = 0.01


class TrafficAPIClient:
    """Fetch traffic data or generate mock data when no API key is provided."""

    def __init__(self, api_key=None, request_timeout=10):
        """
        Initialize the TrafficAPIClient.

        Args:
            api_key (str, optional): API key for the traffic service. Defaults to
                ``os.getenv("TRAFFIC_API_KEY", DEFAULT_TRAFFIC_API_KEY)``. Pass
                an empty string to force mock data.
            request_timeout (int, optional): HTTP request timeout in seconds.
        """
        self.api_key = (
            os.getenv("TRAFFIC_API_KEY", DEFAULT_TRAFFIC_API_KEY)
            if api_key is None
            else api_key
        )
        self.request_timeout = request_timeout
        self.use_mock = not self.api_key
        if self.use_mock:
            logger.warning("No API key provided. Using mock data generator.")

    def fetch_traffic_data(self, location, start_time, end_time):
        """
        Fetch traffic data for a given location and time range.

        Args:
            location (dict): Location identifier, e.g. ``{"lat": 1.0,
                "lng": 2.0}`` or ``{"city": "Doha", "country": "QA"}``.
            start_time (datetime): Start of the time period.
            end_time (datetime): End of the time period.

        Returns:
            pd.DataFrame: Traffic data with timestamp, location_id,
                congestion_score, travel_time_mins, speed_kph, and
                incident_count columns.
        """
        if self.use_mock:
            return self._generate_mock_traffic_data(location, start_time, end_time)

        snapshot = self._fetch_live_traffic_snapshot(location)
        return self._snapshot_to_time_series(snapshot, location, start_time, end_time)

    def _fetch_live_traffic_snapshot(self, location):
        """
        Fetch one live TomTom snapshot for a location.

        TomTom's Flow Segment Data and Incident Details endpoints return current
        traffic conditions, not historical hourly records. The public
        ``fetch_traffic_data`` method preserves this project's time-series shape
        by repeating this snapshot across the requested hourly timestamps.
        """
        lat, lng = self._coordinates(location)
        flow_data = self._fetch_flow_segment_data(lat, lng)
        incident_count = self._fetch_incident_count(location, lat, lng)

        current_speed = self._required_number(flow_data, "currentSpeed")
        free_flow_speed = self._required_number(flow_data, "freeFlowSpeed")
        current_travel_time = self._required_number(flow_data, "currentTravelTime")

        return {
            "congestion_score": self._calculate_congestion_score(
                current_speed,
                free_flow_speed,
            ),
            "travel_time_mins": current_travel_time / 60,
            "speed_kph": current_speed,
            "incident_count": incident_count,
        }

    def _fetch_flow_segment_data(self, lat, lng):
        response = self._tomtom_get_json(
            "/4/flowSegmentData/absolute/10/json",
            {
                "point": f"{lat},{lng}",
                "unit": "KMPH",
            },
        )
        flow_data = response.get("flowSegmentData")
        if not isinstance(flow_data, dict):
            raise ValueError("TomTom Flow Segment Data response missing flowSegmentData")
        return flow_data

    def _fetch_incident_count(self, location, lat, lng):
        response = self._tomtom_get_json(
            "/5/incidentDetails",
            {
                "bbox": self._incident_bbox(location, lat, lng),
                "fields": TOMTOM_INCIDENT_FIELDS,
                "language": "en-GB",
                "timeValidityFilter": "present",
            },
        )
        incidents = response.get("incidents", [])
        if not isinstance(incidents, list):
            raise ValueError("TomTom Incidents response contains invalid incidents data")
        return len(incidents)

    def _tomtom_get_json(self, path, params):
        params = {**params, "key": self.api_key}
        url = f"{TOMTOM_TRAFFIC_BASE_URL}{path}?{urlencode(params, safe=',{}')}"
        try:
            with urlopen(url, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"TomTom request failed with HTTP status {exc.code} for {path}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"TomTom request failed for {path}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"TomTom response was not valid JSON for {path}") from exc

    def _snapshot_to_time_series(self, snapshot, location, start_time, end_time):
        timestamps = pd.date_range(start=start_time, end=end_time, freq="h")
        location_id = self._location_id(location)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "location_id": location_id,
                "congestion_score": snapshot["congestion_score"],
                "travel_time_mins": snapshot["travel_time_mins"],
                "speed_kph": snapshot["speed_kph"],
                "incident_count": snapshot["incident_count"],
            }
        )

    def _coordinates(self, location):
        if "lat" not in location or "lng" not in location:
            raise ValueError(
                "Live TomTom traffic data requires lat and lng in location."
            )
        return float(location["lat"]), float(location["lng"])

    def _incident_bbox(self, location, lat, lng):
        if "bbox" in location:
            return location["bbox"]

        radius = float(
            location.get("incident_radius_degrees", DEFAULT_INCIDENT_RADIUS_DEGREES)
        )
        return f"{lng - radius},{lat - radius},{lng + radius},{lat + radius}"

    def _required_number(self, data, field):
        value = data.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"TomTom Flow Segment Data response missing numeric {field}")
        return value

    def _calculate_congestion_score(self, current_speed, free_flow_speed):
        if free_flow_speed <= 0:
            raise ValueError("TomTom freeFlowSpeed must be greater than zero")
        congestion_score = 100 * (1 - current_speed / free_flow_speed)
        return float(np.clip(congestion_score, 0, 100))

    def _location_id(self, location):
        if "lat" in location and "lng" in location:
            return f"lat_{location['lat']}_lng_{location['lng']}"
        if "city" in location and "country" in location:
            return f"{location['city']}_{location['country']}"
        return "unknown_location"

    def _generate_mock_traffic_data(self, location, start_time, end_time):
        """
        Generate mock traffic data with daily, weekly, incident, and trend
        patterns.
        """
        logger.info(
            "Generating mock traffic data for location %s from %s to %s",
            location,
            start_time,
            end_time,
        )

        location_id = self._location_id(location)
        timestamps = pd.date_range(start=start_time, end=end_time, freq="h")
        n_hours = len(timestamps)

        hour_of_day = timestamps.hour
        day_of_week = timestamps.dayofweek

        morning_peak = np.maximum(0, 1 - np.abs((hour_of_day - 8) / 4))
        evening_peak = np.maximum(0, 1 - np.abs((hour_of_day - 17) / 3))
        daily_pattern = (morning_peak + evening_peak) * 0.6

        weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
        base_congestion = 20 + 50 * daily_pattern * weekly_pattern

        random_noise = np.random.normal(0, 5, n_hours)
        incident_chance = np.random.random(n_hours) < 0.01
        incident_spike = np.where(
            incident_chance,
            np.random.uniform(20, 40, n_hours),
            0,
        )
        time_trend = np.linspace(0, 10, n_hours)

        congestion_score = base_congestion + random_noise + incident_spike + time_trend
        congestion_score = np.clip(congestion_score, 0, 100)

        travel_time_mins = 20 + (congestion_score / 100) * 80
        speed_kph = 60 - (congestion_score / 100) * 40
        incident_count = incident_chance.astype(int)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "location_id": location_id,
                "congestion_score": congestion_score,
                "travel_time_mins": travel_time_mins,
                "speed_kph": speed_kph,
                "incident_count": incident_count,
            }
        )

        logger.info("Generated %s rows of mock traffic data", len(df))
        return df

    def fetch_multiple_locations(self, locations, start_time, end_time):
        """
        Fetch traffic data for multiple locations.

        Returns:
            dict: Mapping of location_id to a DataFrame of traffic data.
        """
        logger.info("Fetching traffic data for %s locations", len(locations))
        data = {}
        for location in locations:
            df = self.fetch_traffic_data(location, start_time, end_time)
            data[self._location_id(location)] = df
        return data
