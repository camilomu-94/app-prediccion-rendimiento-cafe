# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 12:21:07 2026

@author: cmunozo
"""
from os import stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.datasets import fetch_openml
from joblib import load

import warnings

class all_data_frames:
    def __init__(self, seed = 11, test_size = 0.15):
        '''self.datasets = {
            "california_housing": self.load_california_housing,
            "boston_housing": self.load_boston_housing,
            "diabetes": self.load_diabetes,
            "wine_quality": self.load_wine_quality,
            "energy_efficiency": self.load_energy_efficiency,
            "concrete_strength": self.load_concrete_strength,
            "bike_sharing": self.load_bike_sharing,
            "airfoil_self_noise": self.load_airfoil_self_noise,
            "forest_fires": self.load_forest_fires,
            "superconductivity": self.load_superconductivity,
            "cafe": self.load_cafe,
            "cafe_new": self.load_cafe_new,
        }'''
        self.datasets = {
            #"california_housing": self.load_california_housing,
            #"boston_housing": self.load_boston_housing,
            #"diabetes": self.load_diabetes,
            #"wine_quality": self.load_wine_quality,
            #"energy_efficiency": self.load_energy_efficiency,
            #"concrete_strength": self.load_concrete_strength,
            #"bike_sharing": self.load_bike_sharing,
            #"airfoil_self_noise": self.load_airfoil_self_noise,
            #"forest_fires": self.load_forest_fires,
            #"superconductivity": self.load_superconductivity,
            "cafe": self.load_cafe,
            #"cafe_new": self.load_cafe_new,
        }
    def load_dataset(self, name, seed = 11, test_size = 0.10):
        warnings.filterwarnings("ignore")
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(self.datasets.keys())}")
        # Load dataset and split into X and y
        self.datasets[name]()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed, shuffle=True)

    def load_california_housing(self):
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="MedHouseVal")).to_numpy()
        self.y = (df["MedHouseVal"]).to_numpy()

    def load_boston_housing(self):
        data = fetch_openml(name="Boston", version=1, as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="MEDV")).to_numpy()
        self.y = (df["MEDV"]).to_numpy()

    def load_diabetes(self):
        data = load_diabetes(as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="target")).to_numpy()
        self.y = (df["target"]).to_numpy()

    def load_wine_quality(self):
        data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
        df = data.frame
        df["class"] = pd.to_numeric(df["class"], errors="coerce")

        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="class")).to_numpy()
        self.y = (df["class"]).to_numpy()

    def load_energy_efficiency(self):
        data = fetch_openml(name="energy_efficiency", version=1, as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="Y1")).to_numpy()
        self.y = (df["Y1"]).to_numpy()

    def load_concrete_strength(self):
        data = fetch_openml(name="Concrete_Data", version=1, as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="Concrete compressive strength(MPa. megapascals)")).to_numpy()
        self.y = (df["Concrete compressive strength(MPa. megapascals)"]).to_numpy()

    def load_bike_sharing(self):
        data = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True)
        df = data.frame
        df = pd.get_dummies(df, columns=["holiday", "workingday", "season", "weather"], drop_first=True)
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="count")).to_numpy()
        self.y = (df["count"]).to_numpy()

    def load_airfoil_self_noise(self):
        data = fetch_openml(name="airfoil_self_noise", version=1, as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="pressure")).to_numpy()
        self.y = (df["pressure"]).to_numpy()

    def load_forest_fires(self):
        #!pip install ucimlrepo
        #from ucimlrepo import fetch_ucirepo

        forest_fires = fetch_ucirepo(id=162)
        X = forest_fires.data.features

        # Diccionario para mapear los meses a nÃºmeros (por las tres primeras letras en inglÃ©s)
        meses_map = {
            'jan': 1, 'feb': 2,
            'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10,
            'nov': 11, 'dec': 12
        }

        # Mapeo de dÃ­as de la semana en inglÃ©s con sus valores numÃ©ricos
        dias_map = {
            'mon': 1, 'tue': 2,
            'wed': 3, 'thu': 4,
            'fri': 5, 'sat': 6,
            'sun': 7
        }

        # Convertir las columnas "mes" y "dÃ­a" en nÃºmeros
        X['month'] = X['month'].map(meses_map)
        X['day'] = X['day'] .map(dias_map)

        y = forest_fires.data.targets
        df = pd.concat([X, y], axis=1)

        self.feature_names = df.columns[:-1]
        self.X = (X).to_numpy()
        self.y = (y['area']).to_numpy()
        #return X, y['area']

    def load_superconductivity(self):
        data = fetch_openml(name="superconduct", version=1, as_frame=True)
        df = data.frame
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="critical_temp")).to_numpy()
        self.y = (df["critical_temp"]).to_numpy()

    def load_cafe(self):
        #dir = r'C:\Users\cmunozo\PycharmProjects\TRABAJO-Grado-2025\resultados\Coffee pests and diseases survey, Costa Rica, 2002-2003, CIRAD, Jacques Avelino(4).xlsx'
        #dir = 'C:/Users/cmunozo/PycharmProjects/TRABAJO-Grado-2025/resultados/Coffee pests and diseases survey, Costa Rica, 2002-2003, CIRAD, Jacques Avelino(4).xlsx'
        dir = 'C:\\Users\\cmunozo\\PycharmProjects\\TRABAJO-Grado-2025\\Coffee pests and diseases survey, Costa Rica, 2002-2003, CIRAD, Jacques Avelino(4).xlsx'
        df = pd.read_excel(open(dir, 'rb'), sheet_name=2)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna()
        df['yield'] = df['Coffee planting density'] * df['Number of fruiting nodes'] * 2 * 000000.1
        # df['yield'] = df['Number of fruiting nodes']

        # columnas_a_eliminar = ['Coffee planting density','Number of fruiting nodes','Plot ID','Latitude','Longitude','Year'] #,'Number of yielding stems'
        columnas_a_eliminar = ['Coffee planting density','Number of fruiting nodes','Plot ID','Latitude','Longitude','Year',
                               'Total Mar-Dec', 'Acidity', 'Distance between rows', 'Number of ALSD lesions per leaf (Mycena citrocolor)']
        df = df.drop(columnas_a_eliminar, axis=1)
        #df = df.dropna()
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="yield")).to_numpy()
        self.y = (df["yield"]).to_numpy()
        
    def load_cafe_new(self):
        dir = 'C:\\Users\\cmunozo\\PycharmProjects\\TRABAJO-Grado-2025\\nuevo-Dataset_vf.xlsx'
        df = pd.read_excel(open(dir, 'rb'))
        df['yield'] = df['Coffee planting density'] * df['Number of fruiting nodes'] * 2 * 000000.1
        ## df['yield'] = df['Number of fruiting nodes']
        columnas_a_eliminar = ['Number of fruiting nodes','Coffee planting density','Latitude','Longitude']
        df = df.drop(columnas_a_eliminar, axis=1)
        self.feature_names = df.columns[:-1]
        self.X = (df.drop(columns="yield")).to_numpy()
        self.y = (df["yield"]).to_numpy()


dataset=all_data_frames()
best_model = load("best_model.joblib")

dataset.load_dataset("cafe", seed=2)
X_train = dataset.X_train
X_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test

                
resultado_prediccion = best_model.predict(X_test)

print(resultado_prediccion)