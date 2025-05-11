# Preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import category_encoders as ce
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

st.title("Preprocessing et modélisation")

st.subheader("Preprocessing Ademe 2014")

@st.cache_data
def load_data():
    ademe_2014 = pd.read_csv("data/mars-2014-complete.csv", encoding="latin1", sep=";")
    return ademe_2014

ademe_2014 = load_data()

prep_code = """
# Liste des valeurs qu'on veut convertir en numerique
ademe_numeric_names = ['puiss_admin_98','puiss_max','conso_urb','conso_exurb','conso_mixte','co2','co_typ_1','hc','nox','hcnox','ptcl','masse_ordma_min','masse_ordma_max']
# Liste des valeurs catégorielle (sans l'index)
ademe_cat_names = ['lib_mrq' ,'lib_mod_doss','lib_mod','dscom','cnit','tvv','cod_cbr','hybride','typ_boite_nb_rapp','champ_v9','date_maj','Carrosserie','gamme']

var_a_corr = ['puiss_max', 'conso_urb', 'conso_exurb', 'conso_mixte', 'co_typ_1', 'hc', 'nox', 'hcnox', 'ptcl']
for col in ademe_2014[var_a_corr]:
    ademe_2014[col]  = ademe_2014[col].str.replace(",", "", regex = False)
ademe_2014[ademe_numeric_names] =   ademe_2014[ademe_numeric_names].astype(float)


ademe_2014.dropna(subset = ['co2'], axis=0, inplace = True)

#suppression trop de NaN: hc, date_maj

ademe_2014 = ademe_2014.drop('hc', axis=1)
ademe_2014 = ademe_2014.drop('date_maj', axis=1)

# suppression mais à voir pour la jointure avec europe: champ_v9
ademe_2014 = ademe_2014.drop('champ_v9', axis=1)

ademe_2014 = ademe_2014.replace({'EE' : 'essence / électrique (rechargeable)', 'EH' : 'essence / électrique (non rechargeable)', 'EL' : 'électrique', 'ES' : 'essence sans plomb 95', 'ES/GN' : 'bicarburation essence / Gaz Naturel Véhicule – GNV (données consommations essence)',
'ES/GP' : 'bicarburation essence / Gaz de Pétrole Liquéfié – GPL (données consommations essence)', 'FE' : 'véhicule E85 (ou superéthanol-E85)', 'GH' : 'gazole / électrique (non rechargeable)', 'GL' : 'gazole / électrique (rechargeable)',
'GN' : 'monocarburation Gaz Naturel Véhicule – GNV', 'GN/ES' : 'bicarburation essence / Gaz Naturel Véhicule – GNV (données consommations GNV)', 'GO' : 'gazole', 'GP/ES' : 'bicarburation essence / Gaz de Pétrole Liquéfié – GPL (données consommations GPL)',
'H2' : 'hydrogène'})
"""

st.code(prep_code, language="python")

# Liste des valeurs qu'on veut convertir en numerique
ademe_numeric_names = ['puiss_admin_98','puiss_max','conso_urb','conso_exurb','conso_mixte','co2','co_typ_1','hc','nox','hcnox','ptcl','masse_ordma_min','masse_ordma_max']
# Liste des valeurs catégorielle (sans l'index)
ademe_cat_names = ['lib_mrq' ,'lib_mod_doss','lib_mod','dscom','cnit','tvv','cod_cbr','hybride','typ_boite_nb_rapp','champ_v9','date_maj','Carrosserie','gamme']

var_a_corr = ['puiss_max', 'conso_urb', 'conso_exurb', 'conso_mixte', 'co_typ_1', 'hc', 'nox', 'hcnox', 'ptcl']
for col in ademe_2014[var_a_corr]:
    ademe_2014[col]  = ademe_2014[col].str.replace(",", "", regex = False)
ademe_2014[ademe_numeric_names] =   ademe_2014[ademe_numeric_names].astype(float)


ademe_2014.dropna(subset = ['co2'], axis=0, inplace = True)

#suppression trop de NaN: hc, date_maj

ademe_2014 = ademe_2014.drop('hc', axis=1)
ademe_2014 = ademe_2014.drop('date_maj', axis=1)

# suppression mais à voir pour la jointure avec europe: champ_v9
ademe_2014 = ademe_2014.drop('champ_v9', axis=1)

ademe_2014 = ademe_2014.replace({'EE' : 'essence / électrique (rechargeable)', 'EH' : 'essence / électrique (non rechargeable)', 'EL' : 'électrique', 'ES' : 'essence sans plomb 95', 'ES/GN' : 'bicarburation essence / Gaz Naturel Véhicule – GNV (données consommations essence)',
'ES/GP' : 'bicarburation essence / Gaz de Pétrole Liquéfié – GPL (données consommations essence)', 'FE' : 'véhicule E85 (ou superéthanol-E85)', 'GH' : 'gazole / électrique (non rechargeable)', 'GL' : 'gazole / électrique (rechargeable)',
'GN' : 'monocarburation Gaz Naturel Véhicule – GNV', 'GN/ES' : 'bicarburation essence / Gaz Naturel Véhicule – GNV (données consommations GNV)', 'GO' : 'gazole', 'GP/ES' : 'bicarburation essence / Gaz de Pétrole Liquéfié – GPL (données consommations GPL)',
'H2' : 'hydrogène'})

st.subheader("Modelisation Ademe 2014 : GradientBoostingRegressor")

modele_code = """
# Define categorical and numerical features
categorical_features = ['Carrosserie', 'gamme', 'cod_cbr', 'lib_mrq', 'hybride']
numerical_features = ['puiss_admin_98', 'puiss_max', 'ptcl', 'masse_ordma_min', 'masse_ordma_max']

# Prepare the features and target
feats = ademe_2014[categorical_features + numerical_features]
target = ademe_2014['co2']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing
# 1. Handle categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', ce.OneHotEncoder(use_cat_names=True, handle_unknown='ignore', handle_missing='value')), # Use category_encoders' OneHotEncoder
    ('imputer_cat', SimpleImputer(strategy='most_frequent')) # Impute missing values in categorical features after OneHotEncoding
])

# 2. Handle numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer_num', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create the GradientBoostingRegressor model
gbr = GradientBoostingRegressor()

# Define the hyperparameter grid for GridSearchCV
# Access the model parameters using 'model__' prefix
param_grid = {
    'model__n_estimators': [50],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__subsample': [0.8, 0.9, 1.0],
}

# Create a pipeline with preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gbr)
])


#    'model__n_estimators': [50, 100, 200],
#    'model__learning_rate': [0.01, 0.1, 0.2],
#     'model__max_depth': [3, 5, 7],
#     'model__min_samples_split': [2, 5, 10],
#    'model__min_samples_leaf': [1, 2, 4],
#    'model__subsample': [0.8, 0.9, 1.0],

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
"""

st.code(modele_code, language="python")

# Define categorical and numerical features
categorical_features = ['Carrosserie', 'gamme', 'cod_cbr', 'lib_mrq', 'hybride']
numerical_features = ['puiss_admin_98', 'puiss_max', 'ptcl', 'masse_ordma_min', 'masse_ordma_max']

# Prepare the features and target
feats = ademe_2014[categorical_features + numerical_features]
target = ademe_2014['co2']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing
# 1. Handle categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', ce.OneHotEncoder(use_cat_names=True, handle_unknown='ignore', handle_missing='value')), # Use category_encoders' OneHotEncoder
    ('imputer_cat', SimpleImputer(strategy='most_frequent')) # Impute missing values in categorical features after OneHotEncoding
])

# 2. Handle numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer_num', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create the GradientBoostingRegressor model
@st.cache_resource
def load_model():
    # Charger un modèle ML ou en entraîner un
    model = GradientBoostingRegressor()
    return model
    
gbr = load_model()

# Define the hyperparameter grid for GridSearchCV
# Access the model parameters using 'model__' prefix
param_grid = {
    'model__n_estimators': [50],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__subsample': [0.8, 0.9, 1.0],
}

# Create a pipeline with preprocessing and the model
@st.cache_resource
def load_pipeline(_gbr,_preprocessor):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', gbr)
    ])
    return model_pipeline
 
model_pipeline = load_pipeline(gbr,preprocessor)


#    'model__n_estimators': [50, 100, 200],
#    'model__learning_rate': [0.01, 0.1, 0.2],
#     'model__max_depth': [3, 5, 7],
#     'model__min_samples_split': [2, 5, 10],
#    'model__min_samples_leaf': [1, 2, 4],
#    'model__subsample': [0.8, 0.9, 1.0],

# Perform GridSearchCV to find the best hyperparameters
@st.cache_resource 
def grid_search_cv(_model_pipeline,_param_grid):
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search

@st.cache_resource 
grid_s = grid_search_cv(model_pipeline,param_grid)

# Print the best parameters and evaluate the model
y_pred = grid_s.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Résultats")

st.write("MSE : {mse}")
st.write("r2 : {r2}")