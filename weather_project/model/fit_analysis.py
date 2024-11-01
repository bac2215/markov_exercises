import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class fitAnalysis():

    def __init__(self):
        self.df_weather_09_to_23 = pd.read_pickle('./weather_project/data/df_weather_09_to_23.pickle')
        self.df_model_2 = pd.read_pickle('./weather_project/data/df_model2.pickle')
        self.df_model_3 = pd.read_pickle('./weather_project/data/df_model_3.pickle')
        self.df_model4 = pd.read_pickle('./weather_project/data/df_model4.pickle')

    def calc_brier_score_model_one(self):
        df = self.df_weather_09_to_23
        df['brier_score'] = (df['rain_prob'] - df['rain_bool'])**2
        ave_brier_score = df['brier_score'].sum()/df.shape[0]
        return ave_brier_score
    
    def calc_brier_score_model_two(self):
        df = self.df_model_2
        df['brier_score'] = (df['rain_pred_prob'] - df['rain_bool'])**2
        ave_brier_score = df['brier_score'].sum()/df.shape[0]
        return ave_brier_score
    
    def calc_brier_score_model_three(self):
        df = self.df_model_3
        df['brier_score'] = (df['rain_prob'] - df['rain_bool'])**2
        ave_brier_score = df['brier_score'].sum()/df.shape[0]
        return ave_brier_score
    
    def calc_brier_score_model_four(self):
        df = self.df_model4
        df['brier_score'] = (df['rain_prob'] - df['rain_bool'])**2
        ave_brier_score = df['brier_score'].sum()/df.shape[0]
        return ave_brier_score

    def build_brier_score_table(self):
        brier_dict = {'model_one':self.calc_brier_score_model_one(),
                      'model_two':self.calc_brier_score_model_three(),
                      'model_three':self.calc_brier_score_model_four(),
                      'model_four':self.calc_brier_score_model_two()}
        
        return brier_dict

fitAnalysis().build_brier_score_table()
