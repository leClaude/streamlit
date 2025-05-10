# Preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

st.title("Preprocessing et modélisation")

st.subtitle("Preprocessing")

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

#suppression tout en NaN : Unnamed: 26 ,Unnamed: 27 ,Unnamed: 28 ,Unnamed: 29
ademe_2014 = ademe_2014.drop('Unnamed: 26', axis=1)
ademe_2014 = ademe_2014.drop('Unnamed: 27', axis=1)
ademe_2014 = ademe_2014.drop('Unnamed: 28', axis=1)
ademe_2014 = ademe_2014.drop('Unnamed: 29', axis=1)

# suppression mais à voir pour la jointure avec europe: champ_v9
ademe_2014 = ademe_2014.drop('champ_v9', axis=1)
"""

st.code(prep_code, language="python")