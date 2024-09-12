import base64
import json

import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from bson import ObjectId
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pytz

@st.cache_resource
def init_client():
    return MongoClient(uri, server_api=ServerApi('1'))

#toggle on for testing
# from temp.keys import key, uri
# key1 = key
# uri1 = uri
#toggle on for production
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
            # Convert to DataFrame for easier rolling calculation
            dfc = pd.DataFrame(past_data)
            dfc.set_index('Date', inplace=True)
            # Calculate 7-day rolling averages excluding the current day
            rolling_avg_7 = \
            dfc[['Calories', 'Fat', 'Carbs', 'Protein', 'Fiber']].rolling(window=7, min_periods=1).mean().iloc[-1]
            # Calculate 30-day rolling averages excluding the current day
            rolling_avg_30 = \
            dfc[['Calories', 'Fat', 'Carbs', 'Protein', 'Fiber']].rolling(window=30, min_periods=1).mean().iloc[-1]
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
#uncomment
            nt.insert_one(todaydata)
            st.write("Done!")

#-----------------daily goals by ChatGPT--------------------------
# Function to create the circular progress chart
def plot_progress(current_value, max_value, color):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    fig.patch.set_facecolor('black')  # Set the figure background color
    ax.set_facecolor('black')  # Set the axes background color
    # Calculate percentage completion
    percent = current_value / max_value

    # Create the outer ring (full circle)
    ax.pie([percent, 1 - percent], startangle=90, colors=[color, 'darkgray'],
           radius=1.2, wedgeprops=dict(width=0.3))

    # Add text to the center
    ax.text(0, 0, f'{int(percent * 100)}%', ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    ax.text(-0.6, -0.4, f'{int(current_value)}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Remove the axes
    ax.set_aspect('equal')
    plt.axis('off')  # Hide the background axis

    return fig


# Example usage in Streamlit:
st.write("Daily Nutritional Progress")
#delete
# st.write(df)
# st.write(df.columns)
# st.write(df["Date"])
# st.write(df['Date'] == today_date)
# st.write(df[df['Date'] == today_date].empty)
if not df[df['Date'] == today_date].empty:
# if df["Date"].astype(str).str.contains(today_date):
    # Input values (you can replace these with your actual data)
    current_calories = float(df[df['Date'] == today_date]["Calories"]) # Calorie intake for the day
    calorie_goal = 3000  # Calorie goal for the day

    current_protein = float(df[df['Date'] == today_date]["Protein"])  # Protein intake for the day
    protein_goal = 200  # Protein goal for the day

    current_fiber = float(df[df['Date'] == today_date]["Fiber"])  # Fiber intake for the day
    fiber_goal = 30  # Fiber goal for the day

    # Colors for each nutrient
    calorie_color = '#9b5de5'  # Purple for calories
    protein_color = '#ff66b3'  # Pink for protein
    fiber_color = '#4cc9f0'  # Blue for fiber

    # Create the charts for calories, protein, and fiber
    fig_calories = plot_progress(current_calories, calorie_goal, calorie_color)
    fig_protein = plot_progress(current_protein, protein_goal, protein_color)
    fig_fiber = plot_progress(current_fiber, fiber_goal, fiber_color)

    # Display all charts side by side in Streamlit
    # col1, col2, col3 = st.columns(3)
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])  # Equally sized columns

    with col1:
        st.write("Calories")
        st.pyplot(fig_calories)

    with col2:
        st.write("Protein")
        st.pyplot(fig_protein)

    with col3:
        st.write("Fiber")
        st.pyplot(fig_fiber)

#---------------plotting by ChatGPT-------------------------
st.write(" ")
st.write("Trends")
# If there's data
if past_data:
    dfp = pd.DataFrame(past_data)
    dfp.set_index('Date', inplace=True)  # Assuming Date field exists and is formatted correctly
    dfp.index = pd.to_datetime(dfp.index)  # Ensure the Date index is in datetime format
else:
    st.error("No data available for the last 30 days.")
    dfp = pd.DataFrame(columns=['Date', 'Cal7', 'Fat7', 'Car7', 'Pro7', 'Fib7'])  # Empty DataFrame for safety

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Set dark background color for the figure and axes
fig.patch.set_facecolor('black')  # Set figure background color
ax1.set_facecolor('black')  # Set axes background color

# Plot Fat, Carbs, Protein (7-day averages) as line graphs
ax1.plot(dfp.index, dfp['Fat7'], label='Fat (g)', color='#ffcc00', linestyle='-', marker='o', markersize=8, alpha=0.8)
ax1.plot(dfp.index, dfp['Car7'], label='Carbs (g)', color='#2a9d8f', linestyle='-', marker='o', markersize=8, alpha=0.8)
ax1.plot(dfp.index, dfp['Pro7'], label='Protein (g)', color='#e63946', linestyle='-', marker='o', markersize=8, alpha=0.8)

# Set labels for the first y-axis
ax1.set_ylabel('Grams (g)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylim([0, 400])

# Customize ticks and grid
ax1.tick_params(axis='x', rotation=45, colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.grid(True, linestyle='--', alpha=0.3)

# Create a second y-axis for Calories (bars)
ax2 = ax1.twinx()
ax2.bar(dfp.index, dfp['Cal7'], label='Calories (kcal)', color='dimgrey', alpha=0.6, width=0.05)
ax2.set_ylim([0, 4500])

# Set y-axis label
ax2.set_ylabel('Calories (kcal)', fontsize=12)

# Set color for second y-axis ticks
ax2.tick_params(axis='y', colors='white')

# Plot Fiber as scatter plot (dots), scaling the size of the dots based on fiber intake
fiber_sizes = dfp['Fib7'] / 45 * 100  # Scaling fiber sizes (0-45g mapped to 0-100 size)
ax1.scatter(dfp.index, dfp['Fib7'], label='Fiber (g)', color='#9b5de5', s=fiber_sizes, alpha=0.7)

# Add legends with transparent backgrounds
ax1.legend(loc='upper left', frameon=False, labelcolor='white')
ax2.legend(loc='upper right', frameon=False, labelcolor='white')

# Tighten layout
plt.tight_layout()

# Ensure background color is saved correctly
st.pyplot(fig)

#---------------------------add a new food item-------------------------------
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





#----------------------------ideas for future------------------------------------
#---------------add a new food with image--------------------
# File uploader for any type of file
# uploaded_file = st.file_uploader("Or, upload an image of the nutrition facts")
#
# # Text input for the prompt
# st.session_state["foodname"] = st.text_input(label="Food name:")
# if st.button(label="Let's go!"):
#     st.session_state["LG"]=True
# if st.session_state.get("LG",False):
#     # Check if both file and prompt are provided
#     if uploaded_file is not None:
#         if st.session_state["foodname"] == None or len(st.session_state["foodname"].strip()) == 0:
#             st.write("Empty input")
#         else:
#             client = OpenAI(
#
#                 api_key=key,
#             )
#             # Read the file's contents (depending on its type, e.g., text)
#             file_contents = uploaded_file.read()
#             file_base64 = base64.b64encode(file_contents).decode('utf-8')
#
#             # Create a message combining the prompt and file content
#             full_prompt = f'''
#             Provide nutrition information based on the uploaded image (base64 encoded), in a JSON dictionary without formatting, with:
#             - "Food" (value = {st.session_state["foodname"]})
#             - "Calories", "Fat", "Carbs", "Protein", and "Fiber" (meaning soluble fiber) in float with 1 decimal place and no other words.
#
#             Image (base64):
#             {file_base64}
#             '''
#
#             # Send the prompt with file content to ChatGPT using the OpenAI API
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "user", "content": full_prompt},
#                 ]
#             )
#             gptresponse = response.choices[0].message.content
#             st.write(gptresponse)
#             st.write("Does this look okay?")
#             if st.button(label="Confirm"):
#                 ni.insert_one(json.loads(gptresponse))
#                 st.write("Done!")
#--------------------------------------------------------------------------------------------------------------