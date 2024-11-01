import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class getWeatherData():
    def __init__(self):
        self.file_location = "./weather_project/data/"
        self.weather_file = 'df_weather.csv'

    def load_weather_data(self):
        df = pd.read_csv(self.file_location+self.weather_file)
        df['Datum'] = pd.to_datetime(df['Datum'],utc=True)
        return df
    

    def load_all_data(self):
        df_weather= self.load_weather_data()
        return df_weather

    def load_summer_data(self):
        df_weather = self.load_all_data()
        df_weather_summer = df_weather[(df_weather['Datum'].dt.month>=6)&(df_weather['Datum'].dt.month<=9)]
        df_weather_summer = df_weather_summer.reset_index(drop=True)
        return df_weather_summer



#df_weather_summer = getWeatherData().load_summer_data()
