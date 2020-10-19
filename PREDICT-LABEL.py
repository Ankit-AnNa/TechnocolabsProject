#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image
pickle_in = open("clustering_model.pkl","rb")
model=pickle.load(pickle_in)
def predict_label(a,s,d):
    df=pd.read_csv('road-accidents.csv')
    df.rename(columns = {'##### LICENSE #####':'a|b|c|d|e'}, inplace = True)
    split_data = df["a|b|c|d|e"].str.split("|")
    data = split_data.to_list()
    names = ["a", "b","c","d","e"]
    new_df = pd.DataFrame(data, columns=names)
    df1=new_df.rename(columns = {'a':'State','b':'Drivers fatal collisions per billion','c':'Percentage Fatal collisions Speeding','d':'Percentage Fatal collisions Alcohol','e':'Percentage Fatal 1st time'})
    X = df1.drop(['State', 'Drivers fatal collisions per billion'], axis=1)
    scaler = StandardScaler()
    X.loc[-1]=[a,s,d]
    X_scaled = scaler.fit_transform(X)
    prediction = model.predict([X_scaled[-1]])
    print(prediction)
    return prediction
def main():
    st.title("Reducing Traffic Mortality")
    sp=st.slider('Percent of drivers involved in fatal collision who were speeding',0,100)
    al=st.slider('Percent of drivers involved in fatal collision who were Alcohol Impaired',0,100)
    pr=st.slider('Percent of drivers involved in fatal collision who had not been involved in any previous accident',0,100)
    result=[]
    if st.button("Predict the cluster for above data"):
        result = predict_label(sp,al,pr)
        st.success('Cluster {}'.format(result[0]))
    st.text("Developed by: ANKIT NAYAK")
    
if __name__ == '__main__':
    main()


# In[ ]:




