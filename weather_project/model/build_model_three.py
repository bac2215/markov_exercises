import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from etl.build_transition_data import buildTransitionData as btd


class buildModelThree:
    # Model One calculates a barometric pressure transition pressure then applies conditional probability of 
    # rain then does the same for temperature and builds an ensembled rain prediction

    def __init__(self):
        self.df_weather_09_to_23 = btd().run_data()
        pressure_trainsition_matrix, temp_transition_matrix = btd().run_build_transition_matrix()
        self.pressure_transition_matrix = pressure_trainsition_matrix
        self.temp_transition_matrix = temp_transition_matrix
        self.cat_list = ['low_to_low',
                    'low_to_normal',
                    'low_to_high',
                    'normal_to_low',
                    'normal_to_normal',
                    'normal_to_high',
                    'high_to_low',
                    'high_to_normal',
                    'high_to_high']

    def build_rain_no_rain_bool(self):
        df = self.df_weather_09_to_23
        df['rain_bool'] = self.df_weather_09_to_23['Precip'].apply(lambda x: 1 if x>0 else 0)
        return df
    
    
    def build_model_barom_state(self, df_weather_09_to_23):
        high_arr =np.empty([1])
        low_arr = np.empty([1])
        normal_arr = np.empty([1])
            
        
        if df_weather_09_to_23['press_state'].iloc[0] == 'normal':
            high_arr[0] = 0.
            normal_arr[0] = 1.
            low_arr[0]=0.
        if df_weather_09_to_23['press_state'].iloc[0] == 'high':
            high_arr[0]=1.
            normal_arr[0]=0.
            low_arr[0]=0.
        if df_weather_09_to_23['press_state'].iloc[0] == 'low':
            high_arr[0]=0.
            normal_arr[0]=0.
            low_arr[0]=1.

        for i in range(1, df_weather_09_to_23.shape[0]):
            prior_high_state = high_arr[i-1]
            prior_normal_state = normal_arr[i-1]
            prior_low_state = low_arr[i-1]

            prior_state_matrix = np.matrix([prior_low_state,prior_normal_state,prior_high_state])

            new_state_matrix = np.matmul(prior_state_matrix,np.matrix(self.pressure_transition_matrix))
            low_arr = np.append(low_arr,float(new_state_matrix[0,0]))
            normal_arr = np.append(normal_arr,float(new_state_matrix[0,1]))
            high_arr = np.append(high_arr,float(new_state_matrix[0,2]))

        df_weather_09_to_23['low_press_model_state'] = pd.Series(low_arr).astype('float')
        df_weather_09_to_23['normal_press_model_state'] = pd.Series(normal_arr).astype('float')
        df_weather_09_to_23['high_press_model_state'] = pd.Series(high_arr).astype('float')

        return df_weather_09_to_23

    def build_model_temp_state(self, df_weather_09_to_23):
        high_arr =np.empty([1])
        low_arr = np.empty([1])
        normal_arr = np.empty([1])
            
        
        if df_weather_09_to_23['temp_state'].iloc[0] == 'normal':
            high_arr[0] = 0.
            normal_arr[0] = 1.
            low_arr[0]=0.
        if df_weather_09_to_23['temp_state'].iloc[0] == 'high':
            high_arr[0]=1.
            normal_arr[0]=0.
            low_arr[0]=0.
        if df_weather_09_to_23['temp_state'].iloc[0] == 'low':
            high_arr[0]=0.
            normal_arr[0]=0.
            low_arr[0]=1.

        for i in range(1, df_weather_09_to_23.shape[0]):
            prior_high_state = high_arr[i-1]
            prior_normal_state = normal_arr[i-1]
            prior_low_state = low_arr[i-1]

            prior_state_matrix = np.matrix([prior_low_state,prior_normal_state,prior_high_state])

            new_state_matrix = np.matmul(prior_state_matrix,np.matrix(self.temp_transition_matrix))
            low_arr = np.append(low_arr,float(new_state_matrix[0,0]))
            normal_arr = np.append(normal_arr,float(new_state_matrix[0,1]))
            high_arr = np.append(high_arr,float(new_state_matrix[0,2]))

        df_weather_09_to_23['low_temp_model_state'] = pd.Series(low_arr).astype('float')
        df_weather_09_to_23['normal_temp_model_state'] = pd.Series(normal_arr).astype('float')
        df_weather_09_to_23['high_temp_model_state'] = pd.Series(high_arr).astype('float')

        return df_weather_09_to_23

    def calc_rain_probability_by_barom_state(self,df_weather_09_to_23):
        rain_gr_by = df_weather_09_to_23.groupby('press_state')['rain_bool'].sum()
        total_days_by_state = df_weather_09_to_23.groupby('press_state')['press_state'].count()
        barom_rain_prob = rain_gr_by/total_days_by_state
        return barom_rain_prob
    
    def calc_rain_probability_by_temp_state(self,df_weather_09_to_23):
        rain_gr_by = df_weather_09_to_23.groupby('temp_state')['rain_bool'].sum()
        total_days_by_state = df_weather_09_to_23.groupby('temp_state')['temp_state'].count()
        temp_rain_prob = rain_gr_by/total_days_by_state
        return temp_rain_prob
    
    def apply_rain_prob_to_probability(self, df_weather_09_to_23, barom_rain_prob, temp_rain_prob):
        df_weather_09_to_23['barom_rain_prob'] = df_weather_09_to_23.apply(
            lambda row: row.low_press_model_state*barom_rain_prob['low']+row.high_press_model_state*barom_rain_prob['high']+row.normal_press_model_state*barom_rain_prob['normal'],
            axis = 1)
        df_weather_09_to_23['temp_rain_prob'] = df_weather_09_to_23.apply(
            lambda row: row.low_temp_model_state*temp_rain_prob['low']+row.high_temp_model_state*temp_rain_prob['high']+row.normal_temp_model_state*temp_rain_prob['normal'],
            axis = 1)
        df_weather_09_to_23['rain_prob'] = 0.5*df_weather_09_to_23['barom_rain_prob'] + 0.5*df_weather_09_to_23['temp_rain_prob']
        return df_weather_09_to_23
    
    def run_build_model(self):
         df_weather_09_to_23 = self.build_rain_no_rain_bool()
         df_weather_09_to_23 = self.build_model_barom_state(df_weather_09_to_23)
         df_weather_09_to_23 = self.build_model_temp_state(df_weather_09_to_23)
         barom_rain_prob = self.calc_rain_probability_by_barom_state(df_weather_09_to_23)
         temp_rain_prob = self.calc_rain_probability_by_temp_state(df_weather_09_to_23)
         df_weather_09_to_23 = self.apply_rain_prob_to_probability(df_weather_09_to_23,barom_rain_prob, temp_rain_prob)
         
         df_weather_09_to_23.to_pickle('./weather_project/data/df_model_3.pickle')

         return True



buildModelThree().run_build_model()

        




