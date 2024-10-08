import base64
import json
import re  # Add this import to help remove units from the response
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
from streamlit_authenticator import Authenticate, LoginError


@st.cache_resource
def init_client():
    return MongoClient(uri, server_api=ServerApi('1'))

#toggle on for testing
from temp.keys import key, uri, imgur_client_id

# toggle on for production
# key = st.secrets["key"]
# uri = st.secrets["uri"]
# imgur_client_id = st.secrets["imgur_client_id"]

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
    # userdata['preauthorized']
)

try:
    auth.login()
except LoginError as e:
    st.error(e)
# st.write(st.session_state["authentication_status"])
if st.session_state.get("authentication_status"):
    past_data = list(nt.find({"User": st.session_state["username"]}).sort("Date", -1).limit(30))
    prd = pr.find()
    prdata=pd.DataFrame(list(prd))
    #timezone
    today_date = datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")
    st.write(datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d"))

    #rolling averages
    ntt = list(nt.find({"$and":[{"User":st.session_state["username"]},{"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=31))}},{"Date":{"$lt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d"))}}]}))
    ntt.sort(key=lambda x:x["Date"])
    ntt = ntt[::-1]

    def roll_avg7(jsonlist, key):
        items = [x[key] for x in jsonlist[:7]]
        return sum(items)/7

    #title
    st.write("Time to get jacked")

    #tracker
    ntd = nt.find({"$and":[{"User":st.session_state["username"]},{"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=5))}}]})
    df = pd.DataFrame(list(ntd))
    df3 = df
    df3['Consumption'] = df3['Consumption'].astype(str)
    st.dataframe(df3.iloc[:, 1:].set_index('Date'))

    #select food
    nid = ni.find({"User":st.session_state["username"]})
    data=pd.DataFrame(list(nid))
    #choose from dropdown
    dropdown=st.selectbox(label="Food",options=data["Food"],index=list(data["Food"]).index(" "))
    #select quantity
    quantity=st.number_input(label="Quantity",min_value=-5000.1,placeholder=" ",value=None)
    #date
    date=st.date_input(label="Date",value=datetime.datetime.now(pytz.timezone("US/Pacific"))).strftime("%Y-%m-%d")

    if st.button(label="Add"):
        if dropdown == " " or quantity==None:
            st.write("Missing info")
        else:
            dftemp = data[data["Food"] == dropdown]
            dftemp[["Calories","Fat","Carbs","Protein","Fiber"]] = dftemp[["Calories","Fat","Carbs","Protein","Fiber"]].apply(lambda x: float(x) * quantity)

            if date in df['Date'].values:
                todaydata=nt.find_one({"$and":[{"User":st.session_state["username"]},{"Date":date}]})
                fooditem=ni.find_one({"$and":[{"User":st.session_state["username"]},{"Food":dropdown}]})
                todaydata["Calories"]+=quantity*fooditem["Calories"]
                todaydata["Fat"]+=quantity*fooditem["Fat"]
                todaydata["Carbs"] +=quantity*fooditem["Carbs"]
                todaydata["Protein"] +=quantity*fooditem["Protein"]
                todaydata["Fiber"] +=quantity*fooditem["Fiber"]
                fooditem["Quantity"]=quantity
                todaydata["Consumption"].append([fooditem["Food"],fooditem["Quantity"]])
                nt.replace_one({"_id":ObjectId(nt.find_one({"Date":date})["_id"])},todaydata)
                st.write("Done!")

            else:
                todaydata={"Date":date}
                fooditem=ni.find_one({"$and":[{"User":st.session_state["username"]},{"Food":dropdown}]})
                todaydata["Calories"]=quantity*fooditem["Calories"]
                todaydata["Fat"]=quantity*fooditem["Fat"]
                todaydata["Carbs"]=quantity*fooditem["Carbs"]
                todaydata["Protein"]=quantity*fooditem["Protein"]
                todaydata["Fiber"]=quantity*fooditem["Fiber"]
                fooditem["Quantity"]=quantity
                todaydata["Consumption"]=[[fooditem["Food"],fooditem["Quantity"]]]

                # # Add rolling averages to today's data with new column names
                todaydata["Cal7"] = roll_avg7(ntt, "Calories")
                todaydata["Fat7"] = roll_avg7(ntt, "Fat")
                todaydata["Car7"] = roll_avg7(ntt, "Carbs")
                todaydata["Pro7"] = roll_avg7(ntt, "Protein")
                todaydata["Fib7"] = roll_avg7(ntt, "Fiber")
                todaydata["User"] = st.session_state["username"]

                # Insert into MongoDB or whatever your storage solution is
                nt.insert_one(todaydata)
                st.write("Done!")

    foodeditor=st.data_editor(data[data["Food"]==dropdown].iloc[:, 1:-1].set_index('Food'))
    if st.button(label="Update"):
        editedfood=json.loads(foodeditor.to_json(orient="records"))[0]
        editedfood["Food"] = dropdown
        editedfood["User"] = st.session_state["username"]
        ni.replace_one({"$and":[{"User":st.session_state["username"]},{"Food":dropdown}]},editedfood)
        st.write("Done!")

    #-----------------daily goals by ChatGPT--------------------------
    def plot_progress(current_value, max_value, color, ax):
        """
        Plots a progress ring indicating the completion percentage.
        If current_value exceeds max_value, the overflow is indicated in a different color.

        Parameters:
        - current_value (float): The current progress value.
        - max_value (float): The maximum value representing 100% completion.
        - color (str): The color for the completed portion of the progress ring.
        - ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
        """
        # Calculate percentage completion
        percent = current_value / max_value
        overflow = percent > 1  # Determine if there's an overflow

        # Define display percent for the pie chart (max 1)
        display_percent = min(percent, 1)

        # Define colors based on overflow
        if overflow:
            # Completed portion in 'color', overflow portion in 'red'
            colors = [color, 'red', 'darkgray']
            sizes = [1, percent - 1, 0]  # Overflow portion
        else:
            colors = [color, 'darkgray']
            sizes = [display_percent, 1 - display_percent]

        # Create the outer ring (progress ring)
        if overflow:
            # Show completed portion and overflow
            ax.pie(sizes[:2], startangle=90, colors=colors[:2],
                   radius=1.2, wedgeprops=dict(width=0.3, edgecolor='white'))
        else:
            # Show only completed portion
            ax.pie(sizes, startangle=90, colors=colors,
                   radius=1.2, wedgeprops=dict(width=0.3, edgecolor='white'))

        # Add text to the center
        percentage_text = f'{int(min(percent, 1) * 100)}%'
        if overflow:
            percentage_text += '↑'  # Indicate overflow
        ax.text(0, 0, percentage_text, ha='center', va='center',
                fontsize=20, fontweight='bold', color='white')
        ax.text(0, -0.3, f'({int(current_value)} / {max_value})',
                ha='center', va='center', fontsize=15, fontweight='bold', color='white')

        # Set background color and remove axis
        ax.set_facecolor('black')  # Set the axes background color
        ax.set_aspect('equal')
        ax.axis('off')  # Hide the background axis

    # Example usage in Streamlit:
    st.write(" ")
    st.write("Daily Nutritional Progress")
    if not df[df['Date'] == today_date].empty:
        Calories_value = prdata.loc[prdata['Name'] == 'Preset', 'Calories'].values[0]
        protein_value = prdata.loc[prdata['Name'] == 'Preset', 'Protein'].values[0]
        Fiber_value = prdata.loc[prdata['Name'] == 'Preset', 'Fiber'].values[0]

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
        #
        # # Adjust layout and display the figure
        plt.subplots_adjust(wspace=0.3)  # Adjust the space between the subplots

        # Display the figure in Streamlit
        st.pyplot(fig)

    #------presets---------
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
    st.write("7-day rolling averages")

    # If there's data
    if past_data:
        dfp = pd.DataFrame(past_data)
        dfp.set_index('Date', inplace=True)  # Assuming Date field exists and is formatted correctly
        dfp.index = pd.to_datetime(dfp.index)  # Ensure the Date index is in datetime format

        if not dfp.empty:  # Ensure DataFrame has data

            # Plotting
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Set dark background color for the figure and axes
            fig.patch.set_facecolor('black')  # Set figure background color
            ax1.set_facecolor('black')  # Set axes background color

            # Plot Fat, Carbs, Protein (7-day averages) as line graphs
            ax1.plot(dfp.index, dfp['Fat7'], label='Fat (g)', color='#ffcc00', linestyle='-', marker='o', markersize=8,
                     alpha=0.8)
            ax1.plot(dfp.index, dfp['Car7'], label='Carbs (g)', color='#2a9d8f', linestyle='-', marker='o',
                     markersize=8, alpha=0.8)
            ax1.plot(dfp.index, dfp['Pro7'], label='Protein (g)', color='#e63946', linestyle='-', marker='o',
                     markersize=8, alpha=0.8)

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
        else:
            st.write("No data available for the last 7 days.")
    else:
        st.write("No data available for the last 30 days.")

    #---------------------------add a new food item-------------------------------
    st.title("Add food to database")
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
                gptfood["User"]=st.session_state["username"]
                ni.insert_one(gptfood)
                st.write("Done!")

    #---------------add a new food with image--------------------
    # Text input for the prompt
    st.title("Scan nutrition facts")
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

                gptresponse = response.choices[0].message.content
                st.write(gptresponse)
                gptdf = st.data_editor(pd.DataFrame([json.loads(gptresponse)]))
                st.write("Does this look okay?")
                gptfood = json.loads(gptdf.to_json(orient="records"))[0]
                if st.button(label="Confirm"):
                    gptfood["User"]=st.session_state["username"]
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

                        gptresponse = response.choices[0].message.content
                        st.write(gptresponse)
                        gptdf = st.data_editor(pd.DataFrame([json.loads(gptresponse)]))
                        st.write("Does this look okay?")
                        gptfood = json.loads(gptdf.to_json(orient="records"))[0]
                        if st.button(label="Confirm"):
                            gptfood["User"]=st.session_state["username"]
                            ni.insert_one(gptfood)
                            st.write("Done!")

                    else:
                        st.write(response_body)
                except:
                    st.write("error from imgur")

    # -----------------------Recipe input by ChatGPT-----------------------------
    # Track whether the "Cook" button was pressed
    if "cook_pressed" not in st.session_state:
        st.session_state["cook_pressed"] = False

    # Input fields for food name and ingredients
    st.title("Add a recipe")
    food_name = st.text_input(label="Enter food name")
    ingredients = st.text_area(label="and ingredients (separated by commas)")

    # Handle the "Cook" button press
    if st.button(label="Cook"):
        if food_name and ingredients:
            st.session_state["cook_pressed"] = True
            ingredients_list = ingredients.split(",")  # Split the ingredients into a list

            # Prompt GPT-4o for both aggregated and individual ingredient nutrition information
            combined_prompt = f'''
            Provide aggregated nutrition information for the dish "{food_name}" based on the following ingredients (with exact amounts):
            {', '.join(ingredients_list)}. Return the total values for:
            - Calories
            - Fat
            - Carbs
            - Protein
            - Fiber

            Ensure that the values are based on the exact quantities given for each ingredient (e.g., 100g of an ingredient should return values for 100g, not per 100g).

            Then, for each ingredient, provide individual values in a JSON dictionary format, including:
            - "Food" (the name of the ingredient from the list)
            - "Calories"
            - "Fat"
            - "Carbs"
            - "Protein"
            - "Fiber"

            Return the entire response as a list of JSON dictionaries. The first dictionary should represent the aggregated nutrition information for the whole dish, and the remaining dictionaries should represent the individual ingredients. No additional formatting or text.
            '''

            client = OpenAI(api_key=key)

            # Request to GPT-4o for aggregated recipe and ingredient information
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": combined_prompt}]
            )

            # Parse the GPT response
            gpt_response = chat_completion.choices[0].message.content
            st.write(gpt_response)

            try:
                # Parse the response data into a list of dictionaries
                response_data = json.loads(gpt_response)


                # Function to clean up units (like 'g') from the values
                def clean_value(value):
                    # If the value is already a number, return it as is
                    if isinstance(value, (int, float)):
                        return value
                    # Use regex to remove any units (like 'g') and convert the value to a float
                    return float(re.sub(r'[^\d.]+', '', value))


                # Clean the first dictionary (aggregated data for the entire dish)
                aggregated_data = pd.DataFrame([{
                    "Food": food_name,  # Place "Food" as the first item
                    "Calories": clean_value(response_data[0]["Calories"]),
                    "Fat": clean_value(response_data[0]["Fat"]),
                    "Carbs": clean_value(response_data[0]["Carbs"]),
                    "Protein": clean_value(response_data[0]["Protein"]),
                    "Fiber": clean_value(response_data[0]["Fiber"])
                }])

                # Store data in session state
                st.session_state["aggregated_data"] = aggregated_data
                st.session_state["response_data"] = response_data  # Keep raw response for Confirm button use

                # Display the aggregated nutrition info (editable table)
                st.write("Total Nutrition Information for the Dish:")
                st.data_editor(aggregated_data)

                # Clean the remaining dictionaries (ingredients data)
                ingredients_data = []
                for ingredient in response_data[1:]:
                    ingredients_data.append({
                        "Food": ingredient["Food"],  # This will now exist for each ingredient
                        "Calories": clean_value(ingredient["Calories"]),
                        "Fat": clean_value(ingredient["Fat"]),
                        "Carbs": clean_value(ingredient["Carbs"]),
                        "Protein": clean_value(ingredient["Protein"]),
                        "Fiber": clean_value(ingredient["Fiber"])
                    })

                # Convert the ingredients data to a DataFrame
                ingredients_df = pd.DataFrame(ingredients_data)

                # Store the ingredients data in session state
                st.session_state["ingredients_data"] = ingredients_df

                # Display individual ingredient nutrition info
                st.write("Ingredients:")
                st.table(ingredients_df)

                st.write("Does this look okay?")

            except json.JSONDecodeError:
                st.error("Failed to parse GPT response. Please try again.")
                st.write("Raw GPT Response:", gpt_response)  # Display raw response for debugging
            except IndexError:
                st.error("The GPT response does not contain the expected data structure.")
        else:
            st.write("Please enter both a food name and ingredients.")

    # Handle the "Confirm" button press
    if st.session_state.get("cook_pressed", False):
        if st.button(label="Confirm"):
            try:
                # Retrieve aggregated data from session state
                aggregated_data = st.session_state["aggregated_data"]
                aggregated_data_dict = aggregated_data.to_dict(orient="records")[0]

                # Ensure session state has the required key for user authentication
                if "authentication_status" in st.session_state and st.session_state["username"]:
                    aggregated_data_dict["User"] = st.session_state["username"]

                    # Insert into MongoDB
                    ni.insert_one(aggregated_data_dict)

                    # Reset the state
                    st.session_state["cook_pressed"] = False
                    st.write("Done!")

                else:
                    st.error("User not authenticated or session state missing.")

            except Exception as e:
                # Catch any exceptions and print them to help debug the issue
                st.error(f"Failed to insert data into MongoDB: {e}")
                st.write("Debug info:", aggregated_data_dict)

    #-------------------direct input by ChatGPT-------------------
    # Input field for food name, calories, fat, carbs, protein, and fiber
    st.title("Direct input")
    input_data = st.text_input(
        label="Enter food name, calories, fat (g), carbs (g), protein (g), fiber (g) separated by commas")

    if st.button(label="Do it!"):
        # Ensure that the input is not empty
        if input_data:
            try:
                # Split the input by commas
                food_name, calories, fat, carbs, protein, fiber = input_data.split(',')

                # Strip spaces from numerical fields and convert them to floats
                calories = float(calories.strip())
                fat = float(fat.strip())
                carbs = float(carbs.strip())
                protein = float(protein.strip())
                fiber = float(fiber.strip())

                # Create a JSON object with the input values
                food_data = {
                    "Food": food_name.strip(),  # Don't strip spaces from the food name
                    "Calories": calories,
                    "Fat": fat,
                    "Carbs": carbs,
                    "Protein": protein,
                    "Fiber": fiber,
                    "User": st.session_state["username"]  # Add the current user
                }

                # Insert the data into the MongoDB collection
                ni.insert_one(food_data)
                st.write("Done!")
            except ValueError:
                st.error("Please ensure the input is in the correct format with valid numerical values.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter the required data.")