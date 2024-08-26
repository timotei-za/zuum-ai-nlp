import requests
import datetime as dt
import pandas as pd
import os
from dotenv import load_dotenv

class EIAFuelSurcharge():
    def __init__(self):
        load_dotenv()
        self.fscpath = "./fscmatrix.xlsx"
        self.api_key = os.getenv("EIA_API_KEY")
        pd.set_option('display.max_colwidth', None) 
        pd.set_option('display.max_rows', None)
        self.surcharge_matrix = pd.read_excel(self.fscpath, skiprows=[0], names=['Minimum Fuel Price (from)', 'Maximum Fuel Price (to)', 'Fuel Surcharge Per Mile'])
        self.data = self.fetch_fuel_prices()
        self.table_data = self.create_matrix_data()

    def fetch_fuel_prices(self):
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=7)

        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')

        url = f"https://api.eia.gov/v2/petroleum/pri/gnd/data/?api_key={self.api_key}&frequency=weekly&data[0]=value&facets[product][]=EPD2D&facets[product][]=EPD2DXL0&facets[product][]=EPD2DM10&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"ERROR WHEN RETRIEVING FROM EIA GOV API")


    def create_matrix_data(self):
        items = self.data['response']['data']

        df = pd.DataFrame(items)

        def find_surcharge(price):
            for i, row in self.surcharge_matrix.iterrows():
                if row['Minimum Fuel Price (from)'] <= price <= row['Maximum Fuel Price (to)']:
                    return row['Fuel Surcharge Per Mile']
            return 0 
        
        df['surcharge'] = df.apply(lambda row: find_surcharge(float(row['value'])), axis=1)

        result_df = df[['series-description', 'value', 'surcharge']]
        result = result_df.to_dict(orient='records')
        return result

    def get_matrix_data(self):
        return self.table_data
    
if __name__ == '__main__':
    eia = EIAFuelSurcharge()
    print(eia.get_matrix_data())