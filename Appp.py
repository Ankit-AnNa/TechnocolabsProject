#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Accidents Prediction App
This app predicts the most accident prone state of USA
""")
st.text("Developed by: ANKIT NAYAK")
st.write('---')
road_accidents=pd.read_csv('road-accidents.csv',sep='|',comment='#')
miles_driven=pd.read_csv('miles-driven.csv',sep='|',comment='#')
df=pd.concat([road_accidents,miles_driven['million_miles_annually']],axis=1)

X_train=df[['perc_fatl_speed', 'perc_fatl_alcohol','perc_fatl_1st_time','drvr_fatl_col_bmiles','million_miles_annually']]
y_train=df['state']
st.sidebar.header('User Input Features')

def user_input_features():
    drvr_fatl_col_bmiles=st.sidebar.slider('Number of Drivers Involved in Fatal Collision',int(X_train.drvr_fatl_col_bmiles.min()),int(X_train.drvr_fatl_col_bmiles.max()),int(X_train.drvr_fatl_col_bmiles.mean()))
    perc_fatl_speed=st.sidebar.slider('Percentage of Drivers Involved in Over-Speeding',int(X_train.perc_fatl_speed.min()),int(X_train.perc_fatl_speed.max()),int(X_train.perc_fatl_speed.mean()))
    perc_fatl_alcohol=st.sidebar.slider('Percentage of Drivers Alcohol Impaired',int(X_train.perc_fatl_alcohol.min()),int(X_train.perc_fatl_alcohol.max()),int(X_train.perc_fatl_alcohol.mean()))
    perc_fatl_1st_time=st.sidebar.slider('Percentage of Drivers Who had Not been in Accidents',int(X_train.perc_fatl_1st_time.min()),int(X_train.perc_fatl_1st_time.max()),int(X_train.perc_fatl_1st_time.mean()))
    million_miles_annually=st.sidebar.slider('Million Miles Annually',int(X_train.million_miles_annually.min()),int(X_train.million_miles_annually.max()),int(X_train.million_miles_annually.mean()))

    data={'Drivers Fatal Collisions per Billion':drvr_fatl_col_bmiles,
    'Percentage Fatal Collisions Speeding':perc_fatl_speed,
    'Percentage Fatal Collisions Alcohol':perc_fatl_alcohol,
    'Percentage Fatal 1st Time':perc_fatl_1st_time,
    'Million Miles Annually':million_miles_annually
    }

    features=pd.DataFrame(data,index=[0])
    return features

df1=user_input_features()
st.write(df1)
st.write('---')
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

prediction=clf.predict(df1)

st.write("""
### FEASIBLE STATES
""")
st.write(prediction)  


# In[ ]:




