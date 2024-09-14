import base64
import json

import requests
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
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate

@st.cache_resource
def init_client():
    return MongoClient(uri, server_api=ServerApi('1'))

#toggle on for testing
# from temp.keys import key, uri, imgur_client_id

#toggle on for production
key = st.secrets["key"]
uri = st.secrets["uri"]
imgur_client_id = st.secrets["imgur_client_id"]

#mongo
client = init_client()

client.admin.command('ping')
db=client.get_database("db1")
ni=db.get_collection("Nutrition Information")
nt=db.get_collection("Nutrition Tracker")
pr = db.get_collection("Presets")
us = db.get_collection("Users")

#authenticate
userdata=us.find()[0]

auth = Authenticate(
    userdata['credentials'],
    userdata['cookie']['name'],
    userdata['cookie']['key'],
    userdata['cookie']['expiry_days'],
    userdata['preauthorized']
)

st.session_state["authresult"] = auth.login('main')
st.write(st.session_state["authresult"])

if st.session_state.get("authresult",(None,False,None))[1]:
    past_data = list(nt.find({"User":st.session_state["authresult"][2]}).sort("Date", -1).limit(30))
    prd = pr.find()
    prdata=pd.DataFrame(list(prd))

    #timezone
    today_date = datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")
    st.write(datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d"))

    #rolling averages
    ntt = nt.find({"$and":[{"User":st.session_state["authresult"][2]},{"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=31))}}]})
    dft = pd.DataFrame(list(ntt))

    def ra7(series):
        rolling_avg_7 = series.rolling(window=7, min_periods=1).mean().shift(1)  # shift to exclude current day
        return rolling_avg_7

    #title
    st.write("Time to get jacked")

    #tracker
    ntd = nt.find({"$and":[{"User":st.session_state["authresult"][2]},{"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=5))}}]})
    df = pd.DataFrame(list(ntd))
    df3 = df
    df3['Consumption'] = df3['Consumption'].astype(str)
    st.dataframe(df3.iloc[:, 1:].set_index('Date'))

    #select food
    nid = ni.find({"User":st.session_state["authresult"][2]})
    data=pd.DataFrame(list(nid))
    dropdown=st.selectbox(label="Food",options=data["Food"],index=list(data["Food"]).index(" "))
    foodeditor=st.data_editor(data[data["Food"]==dropdown].iloc[:, 1:-1].set_index('Food'))
    if st.button(label="Update"):
        editedfood=json.loads(foodeditor.to_json(orient="records"))[0]
        editedfood["Food"] = dropdown
        editedfood["User"] = st.session_state["authresult"][2]
        ni.replace_one({"$and":[{"User":st.session_state["authresult"][2]},{"Food":dropdown}]},editedfood)
        st.write("Done!")
    #select quantity
    quantity=st.number_input(label="Quantity",min_value=-5000.1,placeholder=" ",value=None)

    #choose from dropdown
    if st.button(label="Add"):
        if dropdown == " " or quantity==None:
            st.write("Missing info")
        else:
            dftemp = data[data["Food"] == dropdown]
            dftemp[["Calories","Fat","Carbs","Protein","Fiber"]] = dftemp[["Calories","Fat","Carbs","Protein","Fiber"]].apply(lambda x: float(x) * quantity)

            if today_date in df['Date'].values:
                todaydata=nt.find_one({"$and":[{"User":st.session_state["authresult"][2]},{"Date":today_date}]})
                fooditem=ni.find_one({"$and":[{"User":st.session_state["authresult"][2]},{"Food":dropdown}]})
                todaydata["Calories"]+=quantity*fooditem["Calories"]
                todaydata["Fat"]+=quantity*fooditem["Fat"]
                todaydata["Carbs"] +=quantity*fooditem["Carbs"]
                todaydata["Protein"] +=quantity*fooditem["Protein"]
                todaydata["Fiber"] +=quantity*fooditem["Fiber"]
                fooditem["Quantity"]=quantity
                todaydata["Consumption"].append([fooditem["Food"],fooditem["Quantity"]])
                nt.replace_one({"_id":ObjectId(nt.find_one({"Date":today_date})["_id"])},todaydata)
                st.write("Done!")

            else:
                todaydata={"Date":today_date}
                fooditem=ni.find_one({"$and":[{"User":st.session_state["authresult"][2]},{"Food":dropdown}]})
                todaydata["Calories"]=quantity*fooditem["Calories"]
                todaydata["Fat"]=quantity*fooditem["Fat"]
                todaydata["Carbs"]=quantity*fooditem["Carbs"]
                todaydata["Protein"]=quantity*fooditem["Protein"]
                todaydata["Fiber"]=quantity*fooditem["Fiber"]
                fooditem["Quantity"]=quantity
                todaydata["Consumption"]=[[fooditem["Food"],fooditem["Quantity"]]]

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
                todaydata["User"]=st.session_state["authresult"][2]
                nt.insert_one(todaydata)
                st.write("Done!")

    #-----------------daily goals by ChatGPT--------------------------
    def plot_progress(current_value, max_value, color, ax):
        # Calculate percentage completion
        percent = current_value / max_value

        # Create the outer ring (full circle)
        ax.pie([percent, 1 - percent], startangle=90, colors=[color, 'darkgray'],
               radius=1.2, wedgeprops=dict(width=0.3))

        # Add text to the center
        ax.text(0, 0, f'{int(percent * 100)}%', ha='center', va='center', fontsize=20, fontweight='bold', color='white')
        ax.text(0, -0.3, f'({int(current_value)} / {max_value})', ha='center', va='center', fontsize=15, fontweight='bold', color='white')

        # Set background color and remove axis
        ax.set_facecolor('black')  # Set the axes background color
        ax.set_aspect('equal')
        ax.axis('off')  # Hide the background axis

    # Example usage in Streamlit:
    st.write("Daily Nutritional Progress")
    if not df[df['Date'] == today_date].empty:
        Calories_value = prdata.loc[prdata['Name'] == 'Preset', 'Calories'].values[0]
        protein_value = prdata.loc[prdata['Name'] == 'Preset', 'Protein'].values[0]
        Fiber_value = prdata.loc[prdata['Name'] == 'Preset', 'Fiber'].values[0]
        # if df["Date"].astype(str).str.contains(today_date):
        # Input values (you can replace these with your actual data)
        current_calories = float(df[df['Date'] == today_date]["Calories"]) # Calorie intake for the day
        calorie_goal = Calories_value  # Calorie goal for the day

        current_protein = float(df[df['Date'] == today_date]["Protein"])  # Protein intake for the day
        protein_goal = protein_value  # Protein goal for the day

        current_fiber = float(df[df['Date'] == today_date]["Fiber"])  # Fiber intake for the day
        fiber_goal = Fiber_value  # Fiber goal for the day

        # Colors for each nutrient
        calorie_color = '#9b5de5'  # Purple for calories
        protein_color = '#ff66b3'  # Pink for protein
        fiber_color = '#4cc9f0'  # Blue for fiber

        # Create the charts for calories, protein, and fiber
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
        fig.patch.set_facecolor('black')  # Set the figure background color

        # Plot each progress chart in a different subplot
        plot_progress(current_calories, calorie_goal, calorie_color, axs[0])
        axs[0].set_title("Calories", color='white')

        plot_progress(current_protein, protein_goal, protein_color, axs[1])
        axs[1].set_title("Protein", color='white')

        plot_progress(current_fiber, fiber_goal, fiber_color, axs[2])
        axs[2].set_title("Fiber", color='white')

        # Adjust layout and display the figure
        plt.subplots_adjust(wspace=0.3)  # Adjust the space between the subplots

        # Display the figure in Streamlit
        st.pyplot(fig)

    #--------------presets-------------------------------------
    #-------unused dropdown ver.--------------
    # prdropdown=st.selectbox(label="Change preset",options=prdata["Name"])
    # preditor=st.data_editor(prdata[prdata["Name"] == prdropdown].iloc[:, 1:])
    # if st.button(label="Change"):
    #     editedpr=json.loads(preditor.to_json(orient="records"))[0]
    #     pr.replace_one({"Name":prdropdown},editedpr)
    #     editedpr["Name"] = "Preset"
    #     pr.replace_one({"Name":"Preset"},editedpr)
    #     st.write("Done!")

    #------button ver.---------
    # Create three columns
    col0, col1, col2, col3 = st.columns(4)

    # Add a button in each column
    with col0: st.write("Change preset:")

    with col1:
        if st.button('Active'):
            pr.update_one({"Name":"Preset"},{"$set": {"Calories": 3000}})
            st.write('Done!')

    with col2:
        if st.button('Very Active'):
            pr.update_one({"Name":"Preset"},{"$set": {"Calories": 3500}})
            st.write('Done!')

    with col3:
        if st.button('Not Active'):
            pr.update_one({"Name":"Preset"},{"$set": {"Calories": 2500}})
            st.write('Done!')


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
            if not st.session_state.get("gptresponse",False):
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
                st.session_state["gptresponse"]=chat_completion.choices[0].message.content

            gptdf = st.data_editor(pd.DataFrame([json.loads(st.session_state["gptresponse"])]))
            gptfood = json.loads(gptdf.to_json(orient="records"))[0]
            st.write("Does this look okay?")
            if st.button(label="Confirm"):
                gptfood["User"]=st.session_state["authresult"][2]
                ni.insert_one(gptfood)
                st.write("Done!")

    #----------------------------ideas for future------------------------------------
    #---------------add a new food with image--------------------
    # Text input for the prompt
    st.session_state["foodname"] = st.text_input(label="Or, enter a food name")

    # File uploader for any type of file
    uploaded_file = st.file_uploader(label="and upload an image of the nutrition facts")

    if st.button(label="Let's go!"):
        st.session_state["LG"]=True
    if st.session_state.get("LG",False):
        # Check if both file and prompt are provided
        if uploaded_file is not None:
            # Upload image to Imgur
            url = "https://api.imgur.com/3/image"
            payload = {'type': 'image',
                       'title': 'Simple upload',
                       'description': 'This is a simple image upload in Imgur'}
            files = [
                ('image', ('GHJQTpX.jpeg', uploaded_file, 'image/jpeg'))
            ]
            headers = {
                'Authorization': f'Client-ID {imgur_client_id}'
            }

            if st.session_state.get("response_link",False):
                client = OpenAI(api_key=key)
                # Create the message combining the prompt and file content
                full_prompt = f'''
                                   Provide nutrition information based on the uploaded image, in a JSON dictionary without formatting, with:
                                   - "Food" (value = {st.session_state["foodname"]})
                                   - "Calories", "Fat", "Carbs", "Protein", and "Fiber" (meaning soluble fiber) in float with 1 decimal place and no other words.
                                   '''
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": st.session_state["response_link"],
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                # response = client.chat.completions.create(
                #     model="gpt-4",
                #     messages=[{"role": "user", "content": full_prompt}]
                # )
                gptresponse = response.choices[0].message.content
                st.write(gptresponse)
                gptdf = st.data_editor(pd.DataFrame([json.loads(gptresponse)]))
                st.write("Does this look okay?")
                gptfood = json.loads(gptdf.to_json(orient="records"))[0]
                if st.button(label="Confirm"):
                    gptfood["User"]=st.session_state["authresult"][2]
                    ni.insert_one(gptfood)
                    st.write("Done!")

            else:
                try:
                    response = requests.request("POST", url, headers=headers, data=payload, files=files)
                    response_body=json.loads(response.text)
                    if response_body["status"] ==200:
                        st.write(response_body["data"]["link"])
                        st.session_state["response_link"]=response_body["data"]["link"]

                        client = OpenAI(api_key=key)
                        # Create the message combining the prompt and file content
                        full_prompt = f'''
                                           Provide nutrition information based on the uploaded image, in a JSON dictionary without formatting, with:
                                           - "Food" (value = {st.session_state["foodname"]})
                                           - "Calories", "Fat", "Carbs", "Protein", and "Fiber" (meaning soluble fiber) in float with 1 decimal place and no other words.
                                           '''
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": full_prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": st.session_state["response_link"],
                                            },
                                        },
                                    ],
                                }
                            ],
                            max_tokens=300,
                        )
                        # response = client.chat.completions.create(
                        #     model="gpt-4",
                        #     messages=[{"role": "user", "content": full_prompt}]
                        # )
                        gptresponse = response.choices[0].message.content
                        st.write(gptresponse)
                        gptdf = st.data_editor(pd.DataFrame([json.loads(gptresponse)]))
                        st.write("Does this look okay?")
                        gptfood = json.loads(gptdf.to_json(orient="records"))[0]
                        if st.button(label="Confirm"):
                            gptfood["User"]=st.session_state["authresult"][2]
                            ni.insert_one(gptfood)
                            st.write("Done!")

                    else:
                        st.write(response_body)
                except:
                    st.write("error from imgur")
#--------------------------------------------------------------------------------------------------------------