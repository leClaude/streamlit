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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

st.title("Preprocessing et modélisation")

st.subheader("Preprocessing Ademe 2014")

@st.cache_data
def load_data():
    ademe_2014 = pd.read_csv("data/mars-2014-complete.csv", encoding="latin1", sep=";")
    ademe_2014 = ademe_2014.drop_duplicates()
    df_sample = ademe_2014.sample(frac=0.1, random_state=42)
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
#'conso_urb','conso_exurb','conso_mixte','co_typ_1','nox','hcnox' sont exclues car issue de test après conception (comme notre variable cible)

target = ademe_2014['co2']

categorical_features_hors_hybride = ['Carrosserie', 'gamme', 'cod_cbr', 'lib_mrq']
encoder_hybride = ['hybride']

#categorical_features = categorical_features_hors_hybride + ordinal_encoder_hybride

numerical_features = ['puiss_admin_98','puiss_max','ptcl','masse_ordma_min','masse_ordma_max']
feats = ademe_2014[categorical_features_hors_hybride + encoder_hybride + numerical_features]


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)

# Création du pipeline de transformation et du modèle
# 1. Traitement des colonnes catégorielles


# on définit les encodeur pour pouvoir les injecter au choix dans la pipeline
target_encoder_instance = ce.TargetEncoder(cols=categorical_features_hors_hybride)
onehot_encoder_instance =  OneHotEncoder(sparse_output=False, handle_unknown='ignore')

#hybride est traité à part car 2 valeurs
categorical_transformer_hybride = Pipeline(steps=[
    ('cat_encoder_hybride', onehot_encoder_instance)])

categorical_transformer = Pipeline(steps=[
    ('cat_encoder', onehot_encoder_instance)])

# 2. Traitement des colonnes numériques

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Remplacer les valeurs manquantes par la moyenne
    ('scaler', StandardScaler())  ]) # Normalisation robuste

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features_hors_hybride),
        ('cat_hybride', categorical_transformer_hybride, encoder_hybride),
        ('num', numerical_transformer, numerical_features)
        ])

#  Gradient Boosting:   GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# GradientBoostingRegressor(n_estimators=500, learning_rate: = 0.2,  max_depth = 7,  min_samples_leaf =  1,  min_samples_split =  5, n_estimators = 50, subsample = 0)

modele = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.2,  max_depth = 7,  min_samples_leaf =  1,  min_samples_split =  5, subsample = 0.8, random_state=42)

# 4. Pipeline complet avec préprocessing et modèle
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', modele) ])

# Entraînement du modèle
model_pipeline.fit(X_train, y_train)


# Prédictions et évaluation du modèle
pred_test = model_pipeline.predict(X_test)

# Affichage des scores
print("Score sur le jeu d'entraînement:", round(model_pipeline.score(X_train, y_train),3))
print("Score sur le jeu de test:", round(model_pipeline.score(X_test, y_test),3))

from sklearn.metrics import mean_squared_error
MSE =  mean_squared_error(y_test, pred_test)
print("MSE:",round(mean_squared_error(y_test, pred_test),2))
import math
print("RMSE: ",  round(math.sqrt(MSE),3))
from sklearn.metrics import mean_absolute_error
print("MAE: ",  round(mean_absolute_error(y_test, pred_test),3))

X_train_transformed = model_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=model_pipeline.named_steps['preprocessor'].get_feature_names_out())
X_test_transformed = pd.DataFrame(X_test_transformed, columns=model_pipeline.named_steps['preprocessor'].get_feature_names_out())

reg = modele
reg.fit(X_train_transformed, y_train)
"""

st.code(modele_code, language="python")

#'conso_urb','conso_exurb','conso_mixte','co_typ_1','nox','hcnox' sont exclues car issue de test après conception (comme notre variable cible)

target = ademe_2014['co2']

categorical_features_hors_hybride = ['Carrosserie', 'gamme', 'cod_cbr', 'lib_mrq']
encoder_hybride = ['hybride']

#categorical_features = categorical_features_hors_hybride + ordinal_encoder_hybride

numerical_features = ['puiss_admin_98','puiss_max','ptcl','masse_ordma_min','masse_ordma_max']
feats = ademe_2014[categorical_features_hors_hybride + encoder_hybride + numerical_features]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)



# Création du pipeline de transformation et du modèle
# 1. Traitement des colonnes catégorielles

# on définit les encodeur pour pouvoir les injecter au choix dans la pipeline
target_encoder_instance = ce.TargetEncoder(cols=categorical_features_hors_hybride)
onehot_encoder_instance =  OneHotEncoder(sparse_output=False, handle_unknown='ignore')

#hybride est traité à part car 2 valeurs
categorical_transformer_hybride = Pipeline(steps=[
    ('cat_encoder_hybride', onehot_encoder_instance)])

categorical_transformer = Pipeline(steps=[
    ('cat_encoder', onehot_encoder_instance)])

# 2. Traitement des colonnes numériques

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Remplacer les valeurs manquantes par la moyenne
    ('scaler', StandardScaler())  ]) # Normalisation robuste


# 4. Pipeline complet avec préprocessing et modèle


# Entraînement du modèle
@st.cache_data
def train_model(_X_train, _y_train):
    # preprocessing
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features_hors_hybride),
        ('cat_hybride', categorical_transformer_hybride, encoder_hybride),
        ('num', numerical_transformer, numerical_features)
        ])
    modele = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.2,  max_depth = 7,  min_samples_leaf =  1,  min_samples_split =  5, subsample = 0.8, random_state=42)
    mp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', modele) ])
    mp.fit(X_train, y_train)
    return mp

model_pipeline = train_model(X_train, y_train)

# Prédictions et évaluation du modèle
pred_test = model_pipeline.predict(X_test)

st.subheader("Résultats")

# Affichage des scores
train_score = round(model_pipeline.score(X_train, y_train),3)
st.write(f"Score sur le jeu d'entraînement: **{train_score}**")

test_score = round(model_pipeline.score(X_test, y_test),3)
st.write(f"Score sur le jeu de test: **{test_score}**")

MSE =  round(mean_squared_error(y_test, pred_test),2)
st.write(f"MSE : **{MSE}**")

RMSE = round(math.sqrt(MSE),3)
st.write(f"RMSE: **{RMSE}**")

MAE = round(mean_absolute_error(y_test, pred_test),3)
st.write(f"MAE: **{MAE}**")




def result_graph(_pred_test, _y_test):
    fig = plt.figure(figsize=(5,10))
    plt.scatter(pred_test, y_test, c='green')
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
    plt.xlabel("prediction")
    plt.ylabel("vrai valeur")
    plt.title('GradientBoostingRegressor pour la prédiction')
    return fig

fig_result = result_graph(pred_test,y_test)

st.plotly_chart(fig_result, use_container_width=True)


X_train_transformed = model_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
X_train_transformed_2 = pd.DataFrame(X_train_transformed, columns=model_pipeline.named_steps['preprocessor'].get_feature_names_out())
X_test_transformed_2 = pd.DataFrame(X_test_transformed, columns=model_pipeline.named_steps['preprocessor'].get_feature_names_out())

@st.cache_data
def train_importance(_X_train_transformed, _y_train):
    mod = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.2,  max_depth = 7,  min_samples_leaf =  1,  min_samples_split =  5, subsample = 0.8, random_state=42)
    mod.fit(X_train_transformed, y_train)
    return mod

reg = train_importance(X_train_transformed_2, y_train)

feat_importances = pd.DataFrame(reg.feature_importances_, index=X_train_transformed_2.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

st.dataframe(feat_importances)

def importance_graph():
    fig = plt.figure(figsize = (10,10))
    plt.barh(feat_importances.index, feat_importances['Importance'])
    plt.title("Importance")
    plt.ylabel(feat_importances.index)
    ax.invert_yaxis()
    plt.xlabel("indicateur")
    
    return fig

fig_importance = importance_graph()

st.plotly_chart(fig_importance, use_container_width=True)


