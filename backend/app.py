import logging
import traceback
from flask import Flask, request
from backend.intents import get_eta, get_status, get_location, get_date, get_price, intent_recognizer
from backend.ner import entity_recognizer
from flasgger import Swagger


app = Flask(__name__)
swagger = Swagger(app)
logging.basicConfig(level=logging.DEBUG)

id2label = {
    0: 'Query_for_Shipment_Location',
    1: 'Query_for_Shipment_StartEnd_Date',
    2: 'Query_for_Shipments_sent_in_windowed_time_period',
    3: 'Query_for_Shipment_Status',
    4: 'Query_for_Shipment_ETA',
    5: 'Query_for_Fuel_Price'
}


@app.route("/api")
def hello_world():
    app.logger.debug('Serving the hello endpoint')
    return "<p>Zuum NLP Project Skeleton</p>"


@app.route("/api/query", methods=['GET'])
def user_query():
    try:
        # app.logger.debug(request.args)
        query = request.args.get('query', '')
        intent = id2label[intent_recognizer(query)]
        print(f'\n ***Qwery -> {query} <- with intent -> {intent} <- *** \n')
        try:
            shipment_id = str(entity_recognizer(query))
        except ValueError as e:
            return str(e)

        if intent == 'Query_for_Shipment_Location':
            return get_location(shipment_id)
        elif intent == 'Query_for_Shipment_Status':
            return get_status(shipment_id)
        elif intent in ['Query_for_Shipment_StartEnd_Date', 'Query_for_Shipments_sent_in_windowed_time_period']:
            return get_date(shipment_id)
        elif intent == 'Query_for_Shipment_ETA':
            return get_eta(shipment_id)
        elif intent == 'Query_for_Fuel_Price':
            return get_price(shipment_id)
        else:
            return 'Could not find the intent, please specify what information you would like'

    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        app.logger.error(f'Error user_query: {error_message}')
        return f'Error when processing request. Error: {e}'


@app.route("/api/recognize")
def user_recognize():
    query = request.args.get('query', '')
    intent = request.args.get('intent', '').strip()
    app.logger.debug(f'Bye World :(! {query} {intent}')
    intent = id2label[intent_recognizer(query)]
    app.logger.debug(intent)
    return "<p>Zuum NLP Project Skeleton</p>"


@app.route("/api/ner")
def user_ner():
    query = request.args.get('query', '')
    intent = request.args.get('intent', '').strip()
    app.logger.debug(f'Oooh World :O! {query} {intent}')
    app.logger.debug(entity_recognizer(query))
    return "<p>Zuum NLP Project Skeleton</p>"


if __name__ == "__main__":
    app.run(debug=True)
