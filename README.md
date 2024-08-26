# Zuum AI NLP
This project is an AI chatbot that allows brokers to easily access information about their shipments and carriers.

## Backend
The backend is a Flask app contained in a Python virtual environment (venv). The backend is middleware that runs the
necessary models and handles all API calls.

To start for development, use the following terminal commands.

```
cd backend
python3 -m venv zoom_env
./zoom_env/Scripts/activate
// install all libaries below, will add requirements.txt soon
flask run
```

The following libraries are installed
- Flask
- Numpy
- pandas
- TensorFlow
- PyTorch (temp)
- Geopy
- Transformers

Once all necessary libraries are installed, the venv will be frozen and a requirements file will be added to make installing on local easier.

To run models using the backend, they need to be directly installed onto the local system and stored in the
zuum_env folder that is created. TBD if this remains the case in development. The backend also relies on several
API keys stored in a .env file, which is only provided if you are on the development team. For local development outside
the team, you can generate personal keys for the external APIs.


## Frontend
The frontend is a React SPA (single page application) built with Vite, that reaches the backend through an API call.

To start development, use the following terminal commands.

```
npm i
npm run dev
```

The command `npm i` will install all necessary packages automatically from package.json. For a list of libraries
installed, see package.json. UI is handled using Chakra UI.
