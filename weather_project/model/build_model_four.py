import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from etl.build_transition_data import buildTransitionData as btd


class buildModelFour:
    # Model Four creates a Markov Chain between the state of rain or no rain

    def __init__(self):
        self.df_model4 = btd().run_data()

    def build_rain_no_rain_bool(self):
        df_model = self.df_model4
        df_model['rain_bool'] = df_model['Precip'].apply(lambda x: 1 if x>0 else 0)
        return df_model
    
    def build_rain_transitions(self, df_model):
        rain_to_rain=[0]
        rain_to_no_rain=[0]
        no_rain_to_rain=[0]
        no_rain_to_no_rain=[0]

        for i in range(1,df_model.shape[0]):
            if df_model['rain_bool'].iloc[i]==1:
                if df_model['rain_bool'].iloc[i-1]==1:
                    rain_to_rain.append(1)
                    rain_to_no_rain.append(0)
                    no_rain_to_rain.append(0)
                    no_rain_to_no_rain.append(0)
                elif df_model['rain_bool'].iloc[i-1]==0:
                    rain_to_rain.append(0)
                    rain_to_no_rain.append(0)
                    no_rain_to_rain.append(1)
                    no_rain_to_no_rain.append(0)
            elif df_model['rain_bool'].iloc[i]==0:
                if df_model['rain_bool'].iloc[i-1]==1:
                    rain_to_rain.append(0)
                    rain_to_no_rain.append(1)
                    no_rain_to_rain.append(0)
                    no_rain_to_no_rain.append(0)
                elif df_model['rain_bool'].iloc[i-1]==0:
                    rain_to_rain.append(0)
                    rain_to_no_rain.append(0)
                    no_rain_to_rain.append(0)
                    no_rain_to_no_rain.append(1)

        df_model['rain_to_rain'] = pd.Series(rain_to_rain)
        df_model['rain_to_no_rain'] = pd.Series(rain_to_no_rain)
        df_model['no_rain_to_rain'] = pd.Series(no_rain_to_rain)
        df_model['no_rain_to_no_rain'] = pd.Series(no_rain_to_no_rain)

        return df_model
    
    def build_rain_trans_matrix(self, df_model):
        rain_to_rain = df_model['rain_to_rain'].sum()/(df_model['rain_to_rain'].sum()+df_model['rain_to_no_rain'].sum())
        rain_to_no_rain = df_model['rain_to_no_rain'].sum()/(df_model['rain_to_rain'].sum()+df_model['rain_to_no_rain'].sum())
        no_rain_to_rain = df_model['no_rain_to_rain'].sum()/(df_model['no_rain_to_rain'].sum()+df_model['no_rain_to_no_rain'].sum())
        no_rain_to_no_rain = df_model['no_rain_to_no_rain'].sum()/(df_model['no_rain_to_rain'].sum()+df_model['no_rain_to_no_rain'].sum())

        rain_trans_matrix = np.matrix(np.array([[rain_to_rain,rain_to_no_rain],[no_rain_to_rain,no_rain_to_no_rain]]))

        return rain_trans_matrix
    
    def build_rain_no_rain_prob(self, df_model, rain_trans_matrix):
        rain_init = np.matrix(np.array([df_model['rain_bool'].iloc[0],1-df_model['rain_bool'].iloc[0]]))
        rain_array = rain_init
        for i in range(1, df_model.shape[0]):
            rain_array = np.append(rain_array, np.matrix(np.matmul(rain_array[i-1],rain_trans_matrix)),axis=0)

        df_model['rain_prob'] = pd.DataFrame(rain_array[:,0])
        df_model['no_rain_prob'] = pd.DataFrame(rain_array[:,1])

        return df_model

    
    def run_build_model(self):
        df_model= self.build_rain_no_rain_bool()
        df_model = self.build_rain_transitions(df_model)
        rain_trans_matrix = self.build_rain_trans_matrix(df_model)
        df_model = self.build_rain_no_rain_prob(df_model,rain_trans_matrix)

        df_model.to_pickle('./weather_project/data/df_model4.pickle')
        
        return True



buildModelFour().run_build_model()

        




