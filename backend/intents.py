from backend.zuumapi import get_shipment_data
from backend.externalapi import get_travel_time, get_shipment_location
from backend.fuelsurcharge import EIAFuelSurcharge
import json
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback


load_dotenv(find_dotenv())


def intent_recognizer(user_input):
    tokenizer = AutoTokenizer.from_pretrained("backend/intents_model")
    model = AutoModelForSequenceClassification.from_pretrained("backend/intents_model")
    inputs = tokenizer(user_input, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        print(f'\n*** logits -> {logits} ***')
        predicted_class_id = logits.argmax().item()
        ###
        print(f'*** predicted_class_id DICT_ID -> {predicted_class_id} *** \n')
        ###
    return predicted_class_id


def get_eta(query):
    shipment_json = json.loads(get_shipment_data(query))

    json_data = json.dumps(shipment_json, indent=4)

    print(shipment_json)
    start_longitude = shipment_json['data']['pickUps'][0]['location']['geoLocation']['long']
    start_latitude = shipment_json['data']['pickUps'][0]['location']['geoLocation']['lat']
    end_longitude = shipment_json['data']['dropOffs'][0]['location']['geoLocation']['long']
    end_latitude = shipment_json['data']['dropOffs'][0]['location']['geoLocation']['lat']
    print(start_longitude, start_latitude, '-\n-', end_longitude, end_latitude)
    try:
        travel_time = get_travel_time((float(start_latitude), float(start_longitude)),
                                      (float(end_latitude), float(end_longitude)))
    except KeyError:
        startSt = str(round(start_longitude, 2)) + ', ' + str(round(start_latitude, 2))
        endSt = str(round(end_longitude, 2)) + ', ' + str(round(end_latitude, 2))
        return f'Cannot find the travel time of this shipment with start location {startSt} and end location {endSt}'
    except Exception as e:
        return f'API currently down, please try again later. Error: {e}'
    print(travel_time)
    return f'Your shipment should arrive in {round(travel_time, 1)} hours'


def get_status(query):
    shipment_json = json.loads(get_shipment_data(query))
    return f"The status of this shipment is {shipment_json['data']['status'].lower()}"


def get_date(shipment_id):
    shipment_json = json.loads(get_shipment_data(shipment_id))
    try:
        return (f"The start date of this shipment is {shipment_json['data']['startDate'].lower()} and the end date "
                f"is {shipment_json['data']['endDate'].lower()}")
    except KeyError:
        error_message = traceback.format_exc()
        print(error_message)
        return 'Could not find the start or end date of this shipment.'


def get_location(shipment_id):
    try:
        shipment_data = get_shipment_data(shipment_id)
        if not shipment_data:
            return "Shipment not available"

        shipment_json = json.loads(shipment_data)
        if not shipment_json or 'data' not in shipment_json:
            return "Shipment data not available"

        if ('job' not in shipment_json['data'] or 'driver' not in shipment_json['data']['job'] or 'geoLocation'
                not in shipment_json['data']['job']['driver']):
            return "No driver locations available"

        shipment_geography = shipment_json['data']['job']['driver']['geoLocation']
        if 'lat' not in shipment_geography or 'long' not in shipment_geography:
            return "Lat and lon not available"

        lat, lon = shipment_geography['lat'], shipment_geography['long']
        print(f'\n *** RESPONSE FOR LOCATION QWERY IS {get_shipment_location(lat, lon)}\n')
        return get_shipment_location(lat, lon)

    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        return f"Error when retrieving shipment location: {e}"


def get_price(shipment_id):
    try:
        eiaFuelSurcharge = EIAFuelSurcharge()
        return eiaFuelSurcharge.get_matrix_data()
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        return "Error when retrieving fuel surcharge chart"


if __name__ == '__main__':
    print(intent_recognizer("what is the status of shipment 1234?"))
