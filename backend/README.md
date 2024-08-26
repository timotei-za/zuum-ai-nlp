## Zuum AI NLP Chatbot Backend

.env contains Keys
The backend for this chatbot is built with flask. The project is divided into the following files (reorganized later)

### app.py
This is the source of the Flask app and where endpoints that connect the Flask backend to the React frontend
exists. Currently only one route of substance is used, `/api/query`. This route takes in two parameters, 
query and intent, and based on the intent and query returns a chatbot response. Once connected to the LLM,
this route will only take in the user query.

### intents.py
This is where different intents are processed. Intents are categorized based on the user query using the intent recognizer
model (different branch currently). Some user intents will require external API calls to get useful inforamtion, for example
ETA requires calls to routing/weather APIs beyond what is stored in the company database. As such the recognizer handles these
seperately before feeding the information into context for the text generator.

### zuumapi.py
This is the zuum company API endpoint. This endpoint returns the JSON file containing shipment details, based on the
shipment ID. It also handles the mapping from shipment ID to shipment object ID.

### externalapi.py
This is where external api calls are handled, for example calls to the google maps, weather, and here.com APIs.

### outgen.py
This is a test file to try and get text generation working.

### db.py
This is for testing dbms access


For details on how to run the backend, see the README.md folder stored in the root directory.
