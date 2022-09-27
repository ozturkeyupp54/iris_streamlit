import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image



st.write("# IRIS- Logistic Rgression")
st.image("iris.png")



st.code("import pandas as pd")
st.code("import pandas as pd\nimport numpy as np")

# with st.echo():
#     import pandas as pd
#     import numpy as np
#     df = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})
#     df

import datetime
today=st.date_input("Today is", datetime.datetime.now())

#time input
the_time= st.time_input("the time is")
st.write("## Zemin güzel, hava güzel, tahmin yürütmek için her şey müsait")
#sidebar
# st.sidebar.title("sidebar little")
# st.sidebar.header("sidebat header")

st.sidebar.title("Enter values")
# st.sidebar.header("Sidebar header")
# a=st.sidebar.slider("input",0,5,2,1)
# x=st.sidebar.slider("input2")
# st.write("# sidebar input result")
# st.success(a*x)


st.write("# Dataframes")
df = pd.read_csv("iris.csv", nrows=(100))
st.table(df.head())
st.write(df.head()) #dynamic, you can sort
st.dataframe(df.head())#dynamic, you can sort


import pickle
filename = 'final_model_iris'
model = pickle.load(open(filename, 'rb'))
scaler_iris= pickle.load(open("scaler_iris","rb"))

sl = st.sidebar.number_input("sepal_length(max 10):",step=1.,format="%.2f")
sw = st.sidebar.number_input("sepal_width(max 5):",step=1.,format="%.2f")
pl = st.sidebar.number_input("petal_length(max 10):",step=1.,format="%.2f")
pw = st.sidebar.number_input("petal_width(max 5):",step=1.,format="%.2f")



dict={"sepal_length":sl,"sepal_width":sw,"petal_length":pl,"petal_width":pw}
df= pd.DataFrame.from_dict([dict])
sample_scaled = scaler_iris.transform(df)

st.table(df)

if st.button("Predict"):
    predictions = model.predict(sample_scaled)
    predictions_proba = model.predict_proba(sample_scaled)
    df["pred"] = predictions
    df["pred_proba_setosa"] = predictions_proba[:,0]
    df["pred_proba_versicolor"] = predictions_proba[:,1]
    df["pred_proba_virginica"] = predictions_proba[:,2]
    st.write(predictions[0])

# html_temp = """
# <div style="background-color:tomato;padding:1.5px">
# <h1 style="color:white;text-align:center;">Single Customer </h1>
# </div><br>"""
# st.sidebar(html_temp,unsafe_allow_html=True)