import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from etl.build_transition_data import buildTransitionData as btd


class buildModelOne:
    # Model One calculates a barometric pressure transition pressure then applies conditional probability of 
    # rain given a state.

    def __init__(self):
        self.df_weather_09_to_23 = btd().run_data()
        pressure_transition_matrix, temp_trans_mat = btd().run_build_transition_matrix()
        self.pressure_transition_matrix = pressure_transition_matrix
        self.cat_list = ['barom_low_to_low',
                    'barom_low_to_normal',
                    'barom_low_to_high',
                    'barom_normal_to_low',
                    'barom_normal_to_normal',
                    'barom_normal_to_high',
                    'barom_high_to_low',
                    'barom_high_to_normal',
                    'barom_high_to_high']

    def build_rain_no_rain_bool(self):
        df = self.df_weather_09_to_23
        df['rain_bool'] = self.df_weather_09_to_23['Precip'].apply(lambda x: 1 if x>0 else 0)
        return df
    
    def calc_joint_bools(self, df_weather_09_to_23):
        cat_list = self.cat_list
    
        for cat in cat_list:
            df_weather_09_to_23[cat+"_and_rain"] = df_weather_09_to_23[cat]*df_weather_09_to_23['rain_bool']

        return df_weather_09_to_23
    
    def calc_joint_probs(self, df_weather_09_to_23):
            cat_list = self.cat_list
            
            joint_prob_dict ={}
            for cat in cat_list:
                 joint_prob_dict[cat] = df_weather_09_to_23[cat+"_and_rain"].sum()/df_weather_09_to_23[cat].sum()

            return joint_prob_dict
    
    def build_model_barom_state(self, joint_prob_dict, df_weather_09_to_23):
        cat_list = self.cat_list
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

        df_weather_09_to_23['low_model_state'] = pd.Series(low_arr).astype('float')
        df_weather_09_to_23['normal_model_state'] = pd.Series(normal_arr).astype('float')
        df_weather_09_to_23['high_model_state'] = pd.Series(high_arr).astype('float')

        return df_weather_09_to_23
    
    def calc_rain_probability_by_state(self,df_weather_09_to_23):
        rain_gr_by = df_weather_09_to_23.groupby('press_state')['rain_bool'].sum()
        total_days_by_state = df_weather_09_to_23.groupby('press_state')['press_state'].count()
        rain_prob = rain_gr_by/total_days_by_state
        return rain_prob
    
    def apply_rain_prob_to_probability(self, df_weather_09_to_23, rain_prob):
        df_weather_09_to_23['rain_prob'] = df_weather_09_to_23.apply(
            lambda row: row.low_model_state*rain_prob['low']+row.high_model_state*rain_prob['high']+row.normal_model_state*rain_prob['normal'],
            axis = 1)
        return df_weather_09_to_23
    
    def run_build_model(self):
         df_weather_09_to_23 = self.build_rain_no_rain_bool()
         df_weather_09_to_23 = self.calc_joint_bools(df_weather_09_to_23)
         joint_prob_dict = self.calc_joint_probs(df_weather_09_to_23)
         df_weather_09_to_23 = self.build_model_barom_state(joint_prob_dict,df_weather_09_to_23)
         rain_prob = self.calc_rain_probability_by_state(df_weather_09_to_23)
         df_weather_09_to_23 = self.apply_rain_prob_to_probability(df_weather_09_to_23,rain_prob)

         df_weather_09_to_23.to_pickle('./weather_project/data/df_weather_09_to_23.pickle')

         return df_weather_09_to_23



buildModelOne().run_build_model()

        




