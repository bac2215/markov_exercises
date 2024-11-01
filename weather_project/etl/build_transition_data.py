import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from etl.get_weather_data import getWeatherData as gwd


class buildTransitionData():

    def __init__(self):
        self.barom_transition_tol = 2
        self.temp_transition_tol = 2

    def load_weather_summer_data(self):
        df_weather_summer = gwd().load_summer_data()
        return df_weather_summer
    
    def iso_summer_2008(self, df_weather_summer):
        df_weather_08 = df_weather_summer[df_weather_summer['Datum'].dt.year == 2008].reset_index(drop=True)
        return df_weather_08
    
    def calc_daily_pressure_statistics(self, df_weather_08):
        press_mean = df_weather_08['Pressure'].mean()
        press_sd = df_weather_08['Pressure'].std()
        return press_mean, press_sd
    
    def calc_daily_temperature_statistics(self, df_weather_08):
        temp_mean = df_weather_08['Temp'].mean()
        temp_sd = df_weather_08['Temp'].std()
        return temp_mean, temp_sd
    
    def iso_summers_09_23(self,df_weather_summer):
        df_weather_09_23 = df_weather_summer[(df_weather_summer['Datum'].dt.year>=2009)&(df_weather_summer['Datum'].dt.year<=2023)]
        return df_weather_09_23
    
    def calc_pressure_change_iso_summers_09_23(self,df_weather_09_23,press_mean,press_sd):
        df_weather_09_23['press_z'] = df_weather_09_23['Pressure'].apply(lambda x: (x-press_mean)/press_sd)
        def calc_press_st(pressz):
            if pressz>=self.barom_transition_tol:
                return 'high'
            elif (pressz>=-1*self.barom_transition_tol):
                return 'normal'
            else:
                return 'low'

        df_weather_09_23['press_state'] = df_weather_09_23['press_z'].apply(lambda x: calc_press_st(x))

        return df_weather_09_23
    
    def calc_temp_change_iso_summers_09_23(self,df_weather_09_23,temp_mean,temp_sd):
        df_weather_09_23['temp_z'] = df_weather_09_23['Temp'].apply(lambda x: (x-temp_mean)/temp_sd)

        def calc_press_st(tempz):
            if tempz>=self.temp_transition_tol:
                return 'high'
            elif (tempz>=-1*self.temp_transition_tol):
                return 'normal'
            else:
                return 'low'

        df_weather_09_23['temp_state'] = df_weather_09_23['temp_z'].apply(lambda x: calc_press_st(x))

        return df_weather_09_23
    
    def calc_barom_state_transitions(self,df_weather_09_23):
        df_weather_09_23 = df_weather_09_23.reset_index(drop=True)
        low_to_low=[]
        low_to_normal=[]
        low_to_high=[]
        normal_to_low=[]
        normal_to_normal=[]
        normal_to_high=[]
        high_to_low=[]
        high_to_normal=[]
        high_to_high = []
        for i in range(1,df_weather_09_23.shape[0]):
            press_state_curr = df_weather_09_23['press_state'].loc[i]
            press_state_prior = df_weather_09_23['press_state'].loc[i-1]
            if press_state_prior=="low":
                if press_state_curr=="low":
                    low_to_low.append(1)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if press_state_curr=="normal":
                    low_to_low.append(0)
                    low_to_normal.append(1)
                    low_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if press_state_curr=="high":
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(1)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
            if press_state_prior=="normal":
                if press_state_curr=="low":
                    normal_to_low.append(1)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if press_state_curr=="normal":
                    normal_to_low.append(0)
                    normal_to_normal.append(1)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if press_state_curr=="high":
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(1)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
            if press_state_prior=="high":
                if press_state_curr=="low":
                    high_to_low.append(1)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                if press_state_curr=="normal":
                    high_to_low.append(0)
                    high_to_normal.append(1)
                    high_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                if press_state_curr=="high":
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(1)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
        
        
        df_weather_09_23['barom_low_to_low'] = pd.Series(low_to_low)
        df_weather_09_23['barom_low_to_normal']= pd.Series(low_to_normal)
        df_weather_09_23['barom_low_to_high']= pd.Series(low_to_high)
        df_weather_09_23['barom_normal_to_low']= pd.Series(normal_to_low)
        df_weather_09_23['barom_normal_to_normal']= pd.Series(normal_to_normal)
        df_weather_09_23['barom_normal_to_high']= pd.Series(normal_to_high)
        df_weather_09_23['barom_high_to_low']= pd.Series(high_to_low)
        df_weather_09_23['barom_high_to_normal']= pd.Series(high_to_normal)
        df_weather_09_23['barom_high_to_high']= pd.Series(high_to_high)

        return df_weather_09_23
    
    def calc_temp_state_transitions(self,df_weather_09_23):
        df_weather_09_23 = df_weather_09_23.reset_index(drop=True)
        low_to_low=[]
        low_to_normal=[]
        low_to_high=[]
        normal_to_low=[]
        normal_to_normal=[]
        normal_to_high=[]
        high_to_low=[]
        high_to_normal=[]
        high_to_high = []
        for i in range(1,df_weather_09_23.shape[0]):
            temp_state_curr = df_weather_09_23['temp_state'].loc[i]
            temp_state_prior = df_weather_09_23['temp_state'].loc[i-1]
            if temp_state_prior=="low":
                if temp_state_curr=="low":
                    low_to_low.append(1)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if temp_state_curr=="normal":
                    low_to_low.append(0)
                    low_to_normal.append(1)
                    low_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if temp_state_curr=="high":
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(1)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
            if temp_state_prior=="normal":
                if temp_state_curr=="low":
                    normal_to_low.append(1)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if temp_state_curr=="normal":
                    normal_to_low.append(0)
                    normal_to_normal.append(1)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                if temp_state_curr=="high":
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(1)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(0)
            if temp_state_prior=="high":
                if temp_state_curr=="low":
                    high_to_low.append(1)
                    high_to_normal.append(0)
                    high_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                if temp_state_curr=="normal":
                    high_to_low.append(0)
                    high_to_normal.append(1)
                    high_to_high.append(0)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
                if temp_state_curr=="high":
                    high_to_low.append(0)
                    high_to_normal.append(0)
                    high_to_high.append(1)
                    normal_to_low.append(0)
                    normal_to_normal.append(0)
                    normal_to_high.append(0)
                    low_to_low.append(0)
                    low_to_normal.append(0)
                    low_to_high.append(0)
        
        
        df_weather_09_23['temp_low_to_low'] = pd.Series(low_to_low)
        df_weather_09_23['temp_low_to_normal']= pd.Series(low_to_normal)
        df_weather_09_23['temp_low_to_high']= pd.Series(low_to_high)
        df_weather_09_23['temp_normal_to_low']= pd.Series(normal_to_low)
        df_weather_09_23['temp_normal_to_normal']= pd.Series(normal_to_normal)
        df_weather_09_23['temp_normal_to_high']= pd.Series(normal_to_high)
        df_weather_09_23['temp_high_to_low']= pd.Series(high_to_low)
        df_weather_09_23['temp_high_to_normal']= pd.Series(high_to_normal)
        df_weather_09_23['temp_high_to_high']= pd.Series(high_to_high)

        return df_weather_09_23
    
    def remove_bad_transitions(self, df_weather_09_23):
        df_weather_09_23 = df_weather_09_23.dropna()

        df_weather_09_23['day_ch'] = df_weather_09_23['Datum'].diff(1)

        df_weather_09_23 = df_weather_09_23[df_weather_09_23['day_ch'].dt.days==1]
        return df_weather_09_23
    
    def build_pressure_transition_matrix(self, df_weather_09_23):
        low_to_low = df_weather_09_23['barom_low_to_low'].sum()
        low_to_normal = df_weather_09_23['barom_low_to_normal'].sum()
        low_to_high = df_weather_09_23['barom_low_to_high'].sum()
        lows_out = low_to_high+low_to_low+low_to_normal
        low_to_normal = low_to_normal/lows_out
        low_to_high = low_to_high/lows_out
        low_to_low = low_to_low/lows_out

        normal_to_low = df_weather_09_23['barom_normal_to_low'].sum()
        normal_to_normal = df_weather_09_23['barom_normal_to_normal'].sum()
        normal_to_high = df_weather_09_23['barom_normal_to_high'].sum()
        normals_out = normal_to_high+normal_to_low+normal_to_normal
        normal_to_normal = normal_to_normal/normals_out
        normal_to_high = normal_to_high/normals_out
        normal_to_low = normal_to_low/normals_out

        high_to_low = df_weather_09_23['barom_high_to_low'].sum()
        high_to_normal = df_weather_09_23['barom_high_to_normal'].sum()
        high_to_high = df_weather_09_23['barom_high_to_high'].sum()
        highs_out = high_to_high+high_to_low+high_to_normal
        high_to_normal = high_to_normal/highs_out
        high_to_high = high_to_high/highs_out
        high_to_low = high_to_low/highs_out

        pressure_transition_matrix = np.array([
                                                [low_to_low,low_to_normal,low_to_high],
                                                [normal_to_low,normal_to_normal,normal_to_high],
                                                [high_to_low,high_to_normal,high_to_high]])
        
        return pressure_transition_matrix

    def build_temp_transition_matrix(self, df_weather_09_23):
        low_to_low = df_weather_09_23['temp_low_to_low'].sum()
        low_to_normal = df_weather_09_23['temp_low_to_normal'].sum()
        low_to_high = df_weather_09_23['temp_low_to_high'].sum()
        lows_out = low_to_high+low_to_low+low_to_normal
        low_to_normal = low_to_normal/lows_out
        low_to_high = low_to_high/lows_out
        low_to_low = low_to_low/lows_out

        normal_to_low = df_weather_09_23['temp_normal_to_low'].sum()
        normal_to_normal = df_weather_09_23['temp_normal_to_normal'].sum()
        normal_to_high = df_weather_09_23['temp_normal_to_high'].sum()
        normals_out = normal_to_high+normal_to_low+normal_to_normal
        normal_to_normal = normal_to_normal/normals_out
        normal_to_high = normal_to_high/normals_out
        normal_to_low = normal_to_low/normals_out

        high_to_low = df_weather_09_23['temp_high_to_low'].sum()
        high_to_normal = df_weather_09_23['temp_high_to_normal'].sum()
        high_to_high = df_weather_09_23['temp_high_to_high'].sum()
        highs_out = high_to_high+high_to_low+high_to_normal
        high_to_normal = high_to_normal/highs_out
        high_to_high = high_to_high/highs_out
        high_to_low = high_to_low/highs_out

        temp_transition_matrix = np.array([
                                                [low_to_low,low_to_normal,low_to_high],
                                                [normal_to_low,normal_to_normal,normal_to_high],
                                                [high_to_low,high_to_normal,high_to_high]])
        
        return temp_transition_matrix

    def run_data(self):
        df_weather_summer = self.load_weather_summer_data()
        df_weather_08 = self.iso_summer_2008(df_weather_summer)
        press_mean,press_sd = self.calc_daily_pressure_statistics(df_weather_08)
        temp_mean, temp_sd = self.calc_daily_temperature_statistics(df_weather_08)
        df_weather_09_to_23 = self.iso_summers_09_23(df_weather_summer)
        df_weather_09_to_23 = self.calc_pressure_change_iso_summers_09_23(df_weather_09_to_23,press_mean,press_sd)
        df_weather_09_to_23 = self.calc_temp_change_iso_summers_09_23(df_weather_09_to_23,temp_mean,temp_sd)
        df_weather_09_to_23 = self.calc_barom_state_transitions(df_weather_09_to_23)
        df_weather_09_to_23 = self.calc_temp_state_transitions(df_weather_09_to_23)
        df_weather_09_to_23 = self.remove_bad_transitions(df_weather_09_to_23)
        df_weather_09_to_23 = df_weather_09_to_23.reset_index(drop=True)
        return df_weather_09_to_23
    
    def run_build_transition_matrix(self):
        df_weather_09_to_23 = self.run_data()
        press_transition_matrix = self.build_pressure_transition_matrix(df_weather_09_to_23)
        temp_transition_matrix = self.build_temp_transition_matrix(df_weather_09_to_23)
        return press_transition_matrix, temp_transition_matrix
    
buildTransitionData().run_build_transition_matrix()