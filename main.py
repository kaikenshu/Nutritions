import json

import streamlit as st
import pandas as pd
import datetime

from bson import ObjectId
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pytz

@st.cache_resource
def init_client():
    return MongoClient(uri, server_api=ServerApi('1'))

# from temp.secrets import key, uri

#secrets
# key1 = key
# uri1 = uri
key = st.secrets["key"]
uri = st.secrets["uri"]

client = init_client()

client.admin.command('ping')
print("Pinged your deployment. You successfully connected to MongoDB!")
db=client.get_database("db1")
ni=db.get_collection("Nutrition Information")
nt=db.get_collection("Nutrition Tracker")

today_date = datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")
st.write(datetime.datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d"))

#title
st.write("Time to get jacked")

#tracker
# ntd = nt.find({"Date":{"$gt":str(datetime.date.today()-datetime.timedelta(days=5))}})
ntd = nt.find({"Date":{"$gt":str(datetime.datetime.strptime(today_date,"%Y-%m-%d")-datetime.timedelta(days=5))}})
df = pd.DataFrame(list(ntd))
st.write(df)
# df=st.data_editor(df0)
# if st.button(label="Save"):
#     df.to_csv("df.csv",index=False)

#select food
nid = ni.find({})
data=pd.DataFrame(list(nid))
dropdown=st.selectbox(label="Food",options=data["Food"],index=list(data["Food"]).index(" "))
foodeditor=st.data_editor(data[data["Food"]==dropdown])
if st.button(label="Save"):
    # st.write(foodeditor.to_json(orient="records"))
    editedfood=json.loads(foodeditor.to_json(orient="records"))[0]
    id=editedfood["_id"]
    del editedfood["_id"]
    ni.replace_one({"_id":ObjectId(id)},editedfood)
    st.write("Done!")
#select quantity
quantity=st.number_input(label="Quantity",min_value=0.1,placeholder=" ",value=None)

#choose from dropdown
if st.button(label="Update"):
    if dropdown == " " or quantity==None:
        st.write("Missing info")
    else:
        dftemp = data[data["Food"] == dropdown]
        dftemp[["Calories","Fat","Carbs","Protein","Fiber"]] = dftemp[["Calories","Fat","Carbs","Protein","Fiber"]].apply(lambda x: float(x) * quantity)
        st.write(dftemp)

        if today_date in df['Date'].values:
            todaydata=nt.find_one({"Date":today_date})
            fooditem=ni.find_one({"Food":dropdown})
            todaydata["Calories"]+=quantity*fooditem["Calories"]
            todaydata["Fat"]+=quantity*fooditem["Fat"]
            todaydata["Carbs"] +=quantity*fooditem["Carbs"]
            todaydata["Protein"] +=quantity*fooditem["Protein"]
            todaydata["Fiber"] +=quantity*fooditem["Fiber"]
            fooditem["Quantity"]=quantity
            todaydata["Consumption"].append(fooditem)
            # st.write(pd.DataFrame([todaydata]))
            nt.replace_one({"_id":ObjectId(nt.find_one({"Date":today_date})["_id"])},todaydata)
            st.write(nt.find_one({"Date":today_date}))

        else:
            todaydata={"Date":today_date}
            fooditem=ni.find_one({"Food":dropdown})
            todaydata["Calories"]=quantity*fooditem["Calories"]
            todaydata["Fat"]=quantity*fooditem["Fat"]
            todaydata["Carbs"]=quantity*fooditem["Carbs"]
            todaydata["Protein"]=quantity*fooditem["Protein"]
            todaydata["Fiber"]=quantity*fooditem["Fiber"]
            fooditem["Quantity"]=quantity
            todaydata["Consumption"]=[fooditem]
            nt.insert_one(todaydata)
            st.write(nt.find_one({"Date":today_date}))

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

#Update the nutrition tracker



#         # df.to_csv("df.csv", index=False)
#
# #Uploader
# tracker=st.file_uploader("Upload a tracker",type="csv")
# if tracker is not None:
#     file=open("data.csv","w")
# #currently x.decode() does not support non-English characters
#     file.writelines([x.decode("utf-8") for x in tracker.readlines()])
#     file.close()

