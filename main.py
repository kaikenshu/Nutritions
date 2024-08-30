import streamlit as st
import pandas as pd

st.write("Time to get jacked")
df0=pd.read_csv(filepath_or_buffer="./df.csv")
df=st.data_editor(df0)

if st.button(label="Save"):
    df.to_csv("df.csv",index=False)

data=pd.read_csv("data.csv")
dropdown=st.selectbox(label="Options",options=data["Food"],placeholder=" ")
st.write(data[data["Food"]==dropdown])

quantity=st.number_input(label="Quantity",min_value=1)

if st.button(label="Update"):
    if dropdown==" ":
        st.write("empty string")
    else:
        dftemp=data[data["Food"]==dropdown]
        dftemp["Calories"]=dftemp["Calories"].apply(lambda x:int(x)*quantity)
        st.write(dftemp)
