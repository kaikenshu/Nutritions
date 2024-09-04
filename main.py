from unittest.mock import inplace

import streamlit as st
import pandas as pd
import os
from datetime import datetime

#CSV
today = datetime.now().strftime('%m/%d/%Y')
today_date = pd.to_datetime(today, format='%m/%d/%Y')

#title
st.write("Time to get jacked")

#CSV
df0=pd.read_csv(filepath_or_buffer="./df.csv")
df=st.data_editor(df0)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

if st.button(label="Save"):
    df.to_csv("df.csv",index=False)

#select food
data=pd.read_csv("data.csv")
dropdown=st.selectbox(label="Food",options=data["Food"],placeholder=" ")
st.write(data[data["Food"]==dropdown])

#select quantity
quantity=st.number_input(label="Quantity",min_value=0.01)

if st.button(label="Update"):
    if dropdown == " ":
        st.write("empty string")
    else:
        dftemp = data[data["Food"] == dropdown]
        dftemp[["Calories","Fat","Carbs","Protein","Fiber"]] = dftemp[["Calories","Fat","Carbs","Protein","Fiber"]].apply(lambda x: float(x) * quantity)
        st.write(dftemp)

        if today_date in df['Date'].values:
            dftemp2=df[df["Date"] == today_date]
            dftemp2["Calories"] = dftemp2["Calories"].apply(lambda x: float(x)+float(dftemp["Calories"]))
            dftemp2["Fat"] = dftemp2["Fat"].apply(lambda x: float(x) + float(dftemp["Fat"]))
            dftemp2["Carbs"] = dftemp2["Carbs"].apply(lambda x: float(x) + float(dftemp["Carbs"]))
            dftemp2["Protein"] = dftemp2["Protein"].apply(lambda x: float(x) + float(dftemp["Protein"]))
            dftemp2["Fiber"] = dftemp2["Fiber"].apply(lambda x: float(x) + float(dftemp["Fiber"]))
            df[df["Date"] == today_date]=dftemp2[dftemp2["Date"] == today_date]
            st.write(df)

        else:
            new_row = [
                today_date,
                float(dftemp["Calories"]),
                float(dftemp["Fat"]),
                float(dftemp["Carbs"]),
                float(dftemp["Protein"]),
                float(dftemp["Fiber"]),
            ]
            df.loc[len(df.index)] = new_row
            st.write(df)

        # df.to_csv("df.csv", index=False)

#Uploader
tracker=st.file_uploader("Upload a tracker",type="csv")
if tracker is not None:
    file=open("data.csv","w")
#currently x.decode() does not support non-English characters
    file.writelines([x.decode("utf-8") for x in tracker.readlines()])
    file.close()


"""
To DO list:
incorporate session state wherever needed
storage => getpantry
"""

