import streamlit as st
import pandas as pd

st.write("Time to get jacked")
input1=st.text_input(label="Food")
st.write("Consumed "+input1)
df0=pd.read_csv(filepath_or_buffer="./df.csv")
df=st.data_editor(df0)

if st.button(label="Save"):
    df.to_csv("df.csv",index=False)

