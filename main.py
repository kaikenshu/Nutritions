import json

import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from bson import ObjectId
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pytz

@st.cache_resource
def init_client():
    return MongoClient(uri, server_api=ServerApi('1'))

#toggle
# from temp.keys import key, uri
# key1 = key
# uri1 = uri
key = st.secrets["key"]
uri = st.secrets["uri"]

client = init_client()

client.admin.command('ping')
db=client.get_database("db1")
ni=db.get_collection("Nutrition Information")
nt=db.get_collection("Nutrition Tracker")
past_data = list(nt.find().sort("Date", -1).limit(30))

#timezone
today_date = datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")
st.write(datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d"))

#rolling averages
ntt = nt.find({"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=31))}})
dft = pd.DataFrame(list(ntt))

def ra7(series):
    rolling_avg_7 = series.rolling(window=7, min_periods=1).mean().shift(1)  # shift to exclude current day
    return rolling_avg_7

#title
st.write("Time to get jacked")

#tracker
# ntd = nt.find({"Date":{"$gt":str(datetime.date.today()-datetime.timedelta(days=5))}})
ntd = nt.find({"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=5))}})
df = pd.DataFrame(list(ntd))
df3 = df
df3['Consumption'] = df3['Consumption'].astype(str)
st.dataframe(df3.iloc[:, 1:].set_index('Date'))

#select food
nid = ni.find({})
data=pd.DataFrame(list(nid))
dropdown=st.selectbox(label="Food",options=data["Food"],index=list(data["Food"]).index(" "))
foodeditor=st.data_editor(data[data["Food"]==dropdown].iloc[:, 1:].set_index('Food'))
if st.button(label="Update"):
    editedfood=json.loads(foodeditor.to_json(orient="records"))[0]
    id=editedfood["_id"]
    del editedfood["_id"]
    ni.replace_one({"_id":ObjectId(id)},editedfood)
    st.write("Done!")
#select quantity
quantity=st.number_input(label="Quantity",min_value=0.1,placeholder=" ",value=None)

#choose from dropdown
if st.button(label="Add"):
    if dropdown == " " or quantity==None:
        st.write("Missing info")
    else:
        dftemp = data[data["Food"] == dropdown]
        dftemp[["Calories","Fat","Carbs","Protein","Fiber"]] = dftemp[["Calories","Fat","Carbs","Protein","Fiber"]].apply(lambda x: float(x) * quantity)

        if today_date in df['Date'].values:
            todaydata=nt.find_one({"Date":today_date})
            fooditem=ni.find_one({"Food":dropdown})
            todaydata["Calories"]+=quantity*fooditem["Calories"]
            todaydata["Fat"]+=quantity*fooditem["Fat"]
            todaydata["Carbs"] +=quantity*fooditem["Carbs"]
            todaydata["Protein"] +=quantity*fooditem["Protein"]
            todaydata["Fiber"] +=quantity*fooditem["Fiber"]
            fooditem["Quantity"]=quantity
            todaydata["Consumption"].append([fooditem["Food"],fooditem["Quantity"]])
            # st.write(pd.DataFrame([todaydata]))
            nt.replace_one({"_id":ObjectId(nt.find_one({"Date":today_date})["_id"])},todaydata)
            st.write("Done!")

        else:
            todaydata={"Date":today_date}
            fooditem=ni.find_one({"Food":dropdown})
            todaydata["Calories"]=quantity*fooditem["Calories"]
            todaydata["Fat"]=quantity*fooditem["Fat"]
            todaydata["Carbs"]=quantity*fooditem["Carbs"]
            todaydata["Protein"]=quantity*fooditem["Protein"]
            todaydata["Fiber"]=quantity*fooditem["Fiber"]
            fooditem["Quantity"]=quantity
            todaydata["Consumption"]=[fooditem["Food"],fooditem["Quantity"]]
#-----------------by ChatGPT------------------------
            if past_data:
                # Convert to DataFrame for easier rolling calculation
                df = pd.DataFrame(past_data)
                df.set_index('Date', inplace=True)
                # Calculate 7-day rolling averages excluding the current day
                rolling_avg_7 = \
                df[['Calories', 'Fat', 'Carbs', 'Protein', 'Fiber']].rolling(window=7, min_periods=1).mean().iloc[-1]
                # Calculate 30-day rolling averages excluding the current day
                rolling_avg_30 = \
                df[['Calories', 'Fat', 'Carbs', 'Protein', 'Fiber']].rolling(window=30, min_periods=1).mean().iloc[-1]
                # Add rolling averages to today's data with new column names
                todaydata["Cal7"] = rolling_avg_7["Calories"]
                todaydata["Fat7"] = rolling_avg_7["Fat"]
                todaydata["Car7"] = rolling_avg_7["Carbs"]
                todaydata["Pro7"] = rolling_avg_7["Protein"]
                todaydata["Fib7"] = rolling_avg_7["Fiber"]
                todaydata["Cal30"] = rolling_avg_30["Calories"]
                todaydata["Fat30"] = rolling_avg_30["Fat"]
                todaydata["Car30"] = rolling_avg_30["Carbs"]
                todaydata["Pro30"] = rolling_avg_30["Protein"]
                todaydata["Fib30"] = rolling_avg_30["Fiber"]
            else:
                # If no past data exists, set rolling averages to None (instead of current day's values)
                todaydata["Cal7"] = None
                todaydata["Fat7"] = None
                todaydata["Car7"] = None
                todaydata["Pro7"] = None
                todaydata["Fib7"] = None
                todaydata["Cal30"] = None
                todaydata["Fat30"] = None
                todaydata["Car30"] = None
                todaydata["Pro30"] = None
                todaydata["Fib30"] = None

            nt.insert_one(todaydata)
            st.write("Done!")

#add a new food item
st.session_state["newfood"] = st.text_input(label="Enter new food and quantity")
if st.button(label="Ask"):
    st.session_state["AskB"]=True
if st.session_state.get("AskB",False):
    if st.session_state["newfood"]==None or len(st.session_state["newfood"].strip())==0:
        st.write("Empty input")
    else:
        client = OpenAI(

            api_key=key,
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f'provide nutrition information for {st.session_state["newfood"]} in a json dictionary without formating, with "Food"(value={st.session_state["newfood"]}) and "Calories", "Fat", "Carbs", "Protein" and "Fiber" (meaning soluble fiber) in float with 1 decimal place and no other words'
                }
            ],
            model="gpt-4o",
        )
        gptresponse=chat_completion.choices[0].message.content
        st.write(gptresponse)
        st.write("Does this look okay?")
        if st.button(label="Confirm"):
            ni.insert_one(json.loads(gptresponse))
            st.write("Done!")

#---------------plotting by ChatGPT-------------------------
# Convert MongoDB data into a pandas DataFrame
if past_data:
    dfp = pd.DataFrame(past_data)
    dfp.set_index('Date', inplace=True)  # Assuming Date field exists and is formatted correctly
    dfp.index = pd.to_datetime(dfp.index)  # Ensure the Date index is in datetime format
    # Assuming that the fields 'Cal7', 'Fat7', 'Car7', 'Pro7', and 'Fib7' exist in the documents
else:
    st.error("No data available for the last 30 days.")
    dfp = pd.DataFrame(columns=['Date', 'Cal7', 'Fat7', 'Car7', 'Pro7', 'Fib7'])  # Empty DataFrame for safety
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot Fat, Carbs, Protein (7-day averages) as line graphs
ax1.plot(dfp.index, dfp['Fat7'], label='Fat (g)', color='orange', linestyle='-', marker='o')
ax1.plot(dfp.index, dfp['Car7'], label='Carbs (g)', color='green', linestyle='-', marker='o')
ax1.plot(dfp.index, dfp['Pro7'], label='Protein (g)', color='red', linestyle='-', marker='o')
# Set labels for the first y-axis
ax1.set_ylabel('Grams (g)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', rotation=45)
# Create a second y-axis for Calories (bars)
ax2 = ax1.twinx()
ax2.bar(dfp.index, dfp['Cal7'], label='Calories (kcal)', color='darkblue', alpha=0.6, width=0.7)
ax2.set_ylabel('Calories (kcal)', fontsize=12)
# Plot Fiber as scatter plot (dots), scaling the size of the dots based on fiber intake
fiber_sizes = dfp['Fib7'] / 45 * 100  # Scaling fiber sizes (0-45g mapped to 0-100 size)
ax1.scatter(dfp.index, dfp['Fib7'], label='Fiber (g)', color='purple', s=fiber_sizes, alpha=0.7)
# Show legends for both axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# Show the plot in Streamlit
st.pyplot(fig)

