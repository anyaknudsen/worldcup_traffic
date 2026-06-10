"""
Traffic API client for fetching traffic data.

When no live API key is available, the client generates mock traffic data for
demonstration and tests.
"""

import logging
import os

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrafficAPIClient:
    """Fetch traffic data or generate mock data when no API key is provided."""

    def __init__(self, api_key=None):
        """
        Initialize the TrafficAPIClient.

        Args:
            api_key (str, optional): API key for the traffic service. Defaults
                to ``os.getenv("TRAFFIC_API_KEY")``.
        """
        self.api_key = api_key or os.getenv("TRAFFIC_API_KEY")
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

        raise NotImplementedError(
            "Live API not implemented. Unset TRAFFIC_API_KEY to use mock data."
        )

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
