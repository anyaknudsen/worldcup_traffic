
  """
  traffic_client.py
  Traffic API Client for fetching traffic data.
  Includes mock data generation for demonstration when no live API key is
  available.
  """

  import os
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta
  import logging

  # Set up logging
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -
  %(message)s')
  logger = logging.getLogger(__name__)


  class TrafficAPIClient:
      """
      A client for fetching traffic data from a traffic API.
      If no API key is provided, generates mock data for demonstration.
      """

      def __init__(self, api_key=None):
          """
          Initialize the TrafficAPIClient.

          Args:
              api_key (str, optional): API key for the traffic service.
                                     Defaults to os.getenv('TRAFFIC_API_KEY').
          """
          self.api_key = api_key or os.getenv('TRAFFIC_API_KEY')
          self.use_mock = not self.api_key
          if self.use_mock:
              logger.warning("No API key provided. Using mock data generator.")

      def fetch_traffic_data(self, location, start_time, end_time):
          """
          Fetch traffic data for a given location and time range.

          Args:
              location (dict): Location identifier (e.g., {'lat': float, 'lng':
  float} or {'city': str, 'country': str})
              start_time (datetime): Start of the time period
              end_time (datetime): End of the time period

          Returns:
              pd.DataFrame: DataFrame containing traffic data with columns:
                  - timestamp: datetime
                  - location_id: str (identifier for the location)
                  - congestion_score: float (0-100, higher means more
  congestion)
                  - travel_time_mins: float (estimated travel time in minutes)
                  - speed_kph: float (average speed in km/h)
                  - incident_count: int (number of reported incidents)
          """
          if self.use_mock:
              return self._generate_mock_traffic_data(location, start_time,
  end_time)
          else:
              # In a real implementation, this would call the actual API
              # For example:
              # response = requests.get(
              #     f"https://api.traffic.com/v1/data",
              #     params={
              #         'location': location,
              #         'start_time': start_time.isoformat(),
              #         'end_time': end_time.isoformat()
              #     },
              #     headers={'Authorization': f'Bearer {self.api_key}'}
              # )
              # data = response.json()
              # return pd.DataFrame(data)
              raise NotImplementedError("Live API not implemented. Please
  provide an API key or use mock data.")

      def _generate_mock_traffic_data(self, location, start_time, end_time):
          """
          Generate mock traffic data with realistic patterns.

          Args:
              location (dict): Location identifier
              start_time (datetime): Start time
              end_time (datetime): End time

          Returns:
              pd.DataFrame: Mock traffic data
          """
          logger.info(f"Generating mock traffic data for location {location}
  from {start_time} to {end_time}")

          # Create location ID
          if 'lat' in location and 'lng' in location:
              location_id = f"lat_{location['lat']}_lng_{location['lng']}"
          elif 'city' in location and 'country' in location:
              location_id = f"{location['city']}_{location['country']}"
          else:
              location_id = "unknown_location"

          # Generate hourly timestamps
          timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
          n_hours = len(timestamps)

          # Initialize arrays for our features
          hour_of_day = timestamps.hour
          day_of_week = timestamps.dayofweek  # Monday=0, Sunday=6

          # Base congestion level (0-100)
          # We'll create patterns:
          # 1. Daily pattern: morning rush (7-9 AM), evening rush (4-7 PM)
          # 2. Weekly pattern: lower congestion on weekends
          # 3. Random incidents: occasional spikes
          # 4. Seasonal trends: slight increase over time (for demonstration)

          # Daily pattern: two peaks
          morning_peak = np.maximum(0, 1 - np.abs((hour_of_day - 8) / 4))  #
  Peaks at 8 AM
          evening_peak = np.maximum(0, 1 - np.abs((hour_of_day - 17) / 3))  #
  Peaks at 5 PM
          daily_pattern = (morning_peak + evening_peak) * 0.6  # Scale to 0-0.6

          # Weekly pattern: reduce congestion on weekends (Sat=5, Sun=6)
          weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)  # 30% reduction
  on weekends

          # Combine daily and weekly patterns
          base_congestion = 20 + 50 * daily_pattern * weekly_pattern  # Base
  between 20-70

          # Add random noise and incidents
          random_noise = np.random.normal(0, 5, n_hours)

          # Random incidents: 1% chance of an incident per hour, causing a spike
          incident_chance = np.random.random(n_hours) < 0.01
          incident_spike = np.where(incident_chance, np.random.uniform(20, 40,
  n_hours), 0)

          # Slight upward trend over time (to simulate growing traffic)
          time_trend = np.linspace(0, 10, n_hours)  # Up to 10 points increase
  over the period

          # Combine all factors
          congestion_score = base_congestion + random_noise + incident_spike +
  time_trend
          congestion_score = np.clip(congestion_score, 0, 100)  # Ensure 0-100
  range

          # Derive travel time and speed from congestion score
          # Simple inverse relationship: higher congestion -> longer travel
  time, lower speed
          # Base travel time: 20 minutes (no congestion)
          base_travel_time = 20
          # Congestion adds to travel time: 0-80 extra minutes based on
  congestion score
          travel_time_mins = base_travel_time + (congestion_score / 100) * 80

          # Speed: inversely related to congestion (simplified)
          # Base speed: 60 kph (no congestion), drops to 20 kph at max
  congestion
          speed_kph = 60 - (congestion_score / 100) * 40

          # Incident count: 0 or 1 for simplicity (could be Poisson)
          incident_count = incident_chance.astype(int)

          # Create DataFrame
          df = pd.DataFrame({
              'timestamp': timestamps,
              'location_id': location_id,
              'congestion_score': congestion_score,
              'travel_time_mins': travel_time_mins,
              'speed_kph': speed_kph,
              'incident_count': incident_count
          })

          logger.info(f"Generated {len(df)} rows of mock traffic data")
          return df

      def fetch_multiple_locations(self, locations, start_time, end_time):
          """
          Fetch traffic data for multiple locations.

          Args:
              locations (list): List of location dictionaries
              start_time (datetime): Start time
              end_time (datetime): End time

          Returns:
              dict: Dictionary mapping location_id to DataFrame of traffic data
          """
          logger.info(f"Fetching traffic data for {len(locations)} locations")
          data = {}
          for location in locations:
              df = self.fetch_traffic_data(location, start_time, end_time)
              # Use location_id as key
              loc_id = df['location_id'].iloc[0]
              data[loc_id] = df
          return data


