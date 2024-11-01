import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from etl.build_transition_data import buildTransitionData as btd


class buildModelTwo:
    # Model Two is a naive coinflip each day on whether or not it will rain.

    def __init__(self):
        self.df_model2 = btd().run_data()

    def build_rain_no_rain_bool(self):
        df_model_2 = self.df_model2
        df_model_2['rain_bool'] = df_model_2['Precip'].apply(lambda x: 1 if x>0 else 0)
        return df_model_2
    
    def coin_flip_predict(self, df_model_2):
        df_model_2['rain_pred_prob'] = 0.5
        return df_model_2

    
    def run_build_model(self):
        df_model_2 = self.build_rain_no_rain_bool()
        df_model_2 = self.coin_flip_predict(df_model_2)


        df_model_2.to_pickle('./weather_project/data/df_model2.pickle')
        
        return True



buildModelTwo().run_build_model()

        




