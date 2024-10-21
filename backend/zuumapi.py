import requests
from dotenv import load_dotenv, find_dotenv
import os
import json


load_dotenv(find_dotenv())


def get_shipment_data(shipment_id, save=True):
    header_data = {'accept': 'application/json', 'x-access-token': os.environ.get('ZUUM_TOKEN')}
    # endpoint = 'https://stage-ai-api.zuumapp.com/shipments/' + shipment_id + '/' + os.environ.get('TENANT_ID')
    endpoint = 'https://stage-ship-api.zuumapp.com/shipments/ai/' + shipment_id + '/' + os.environ.get('TENANT_ID')
    # json_data = json.dumps(endpoint, indent=4)

    shipment_statuses = requests.get(endpoint, headers=header_data)

    print(f'\n Response -> {shipment_statuses}')

    json_data = json.dumps(shipment_statuses.json(), indent=4)
    if save:
        with open("response.json", "w") as file:
            file.write(json_data)
    return json_data
