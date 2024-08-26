import requests
from dotenv import load_dotenv, find_dotenv
import os
import json
from geopy.distance import geodesic
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


def get_weather(query_type, location, one_observation, hourly_date=None):
    api_key = os.environ.get('HERE_TOKEN')
    base_url = f'https://weather.hereapi.com/v3/report?products=observation&apiKey={api_key}&oneObservation={one_observation}'
    if query_type == '1':
        base_url += f"&q={location}"
    else:
        lat, lon = location.split(',')
        base_url += f"&location={lat},{lon}"
    if hourly_date:
        base_url += f"&hourlyDate={hourly_date}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return "Error when retrieving data!"


def get_weather(location, api_key):
    base_url = f'https://weather.hereapi.com/v3/report?products=observation&location={location}&apiKey={api_key}&oneObservation=True'
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        description = data['places'][0]['observations'][0]['description']
        return description.lower()
    else:
        return "Error when retrieving data!"


def get_travel_time(start_coords, end_coords):
    here_api_key = os.environ.get('HERE_TOKEN')
    base_url = f"https://router.hereapi.com/v8/routes?transportMode=truck&origin={start_coords[0]},{start_coords[1]}&destination={end_coords[0]},{end_coords[1]}&return=summary&apikey={here_api_key}"
    response = requests.get(base_url)

    if response.status_code == 200:
        data = response.json() 
        if 'routes' not in data:
            print("No route found.")
            return None
        duration = data['routes'][0]['sections'][0]['summary']['duration'] / 3600 
        return duration
    else:
        return "Error when retrieving data!"


def get_shipment_location(lat, lon):
    api_key = os.environ.get('GOOGLE_MAPS_TOKEN')
    api_url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}'
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        location = data['results'][0]['formatted_address']
        return location
    else: 
        return "Error when retrieving location"


def adjust_for_weather_conditions(weather_conditions, travel_time):
    weather_conditions = weather_conditions.lower()

    if 'rain' in weather_conditions:
        travel_time *= 1.2  
    if 'heavy rain' in weather_conditions:
        travel_time *= 1.4  
    if 'fog' in weather_conditions:
        travel_time *= 1.3  
    if 'snow' in weather_conditions:
        travel_time *= 1.5  
    if 'ice' in weather_conditions:
        travel_time *= 1.6  
    if 'wind' in weather_conditions:
        travel_time *= 1.1

    return travel_time
