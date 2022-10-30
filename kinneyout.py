# importing the required module
from codecs import ignore_errors
from select import select
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from requests import session
import streamlit as st

# initial page config
st.set_page_config(
     page_title="Kinney:Out",
     layout="wide",
     initial_sidebar_state="expanded",
)

#converting a list to a string for filenames
def listToString(s):
    # initialize an empty string
    str1 = ""
    for x in s:
        str1 = str1 + "__" + x
    return str1

#parse filename to get vehicle descriptor -- return a dict with names
def parse_filename(name):
    #removing .csv and splitting name based on '__'
    rem_type = name.split('.csv')
    str_name = rem_type[0].split('__')

    #concatting the seperated strings to give the correct vehicle descriptors
    model_template = (listToString(str_name[0:3]))[2:]
    test_variant = (listToString(str_name[3:6]))[2:]
    test_bench = (listToString(str_name[6:8]))[2:]
    test_load_case = (listToString(str_name[8:]))[2:]

    return {"mt": model_template, "tv": test_variant, "tb": test_bench, "tlc": test_load_case}

#get all datasets based on test bench name passed in paramenter, return list
def get_axis(tb):
    temp_list = [ x for x in all_df.index.tolist() if x[0] == tb ]
    axis = [x[1] for x in temp_list]
    return axis

#get row based on test bench name and dataset passed in parameter, return row
def get_data(df, tb, x, y):
    df1 = df.loc[tb].loc[x]
    df2 = df.loc[tb].loc[y]
    return pd.concat([df1, df2], axis=1)

def axis_set(x, x2, xfunc):
    return [x, x2, xfunc]

#create dict based on graph selection and options
def update_dict(df_name, x, y):
    return {"df_name":df_name, "x_axis":x[0], "x_secondary":x[1], "x_math":x[2], "y_axis":y[0], "y_secondary":y[1], "y_math":y[2]}
    
#get df
#str(df_list[df_name_list.index(selected_graph)])

st.title("Welcome to the Kinney:Out Results Viewer")
uploaded_files = st.file_uploader("Please select datasets to be graphed.", accept_multiple_files=True)

#create a df that is a concationation of all .csv files
all_df = pd.DataFrame()

#create a list to hold all datasets
df_list = []
df_name_list = []

#check whether user has uploaded any files
if len(uploaded_files) != 0 and len(uploaded_files) <= 5:
    #if so, run through files and run rest of code
    for uploaded_file in uploaded_files:
        #read csv file and convert to df
        df = pd.read_csv(uploaded_file)
        #parse the csv file name into a list of four segments
        filenames = parse_filename(uploaded_file.name)

        #add filename to list to choose from
        df_name_list.append(filenames['tv'])

        #create a temporary df for manipulation
        temp_df = df

        #clean up the testbench name
        temp_df.iloc[:,0] = temp_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")

        #add another column based on file and add it to the index to create a MultiIndex
        temp_df = temp_df.assign(test = filenames["tv"])
        temp_df.set_index(['test','time'], inplace=True)
        temp_df.rename(index={'times':'dataset'})

        #add tempoary df to df holding all df's
        all_df = pd.concat([all_df, temp_df])

        

    st.write(all_df)
    selected_graph = st.sidebar.selectbox("Select graph to work on", df_name_list, key="data_select")
    #graph_title = st.sidebar.text_input("Graph Title", selected_graph, key="title")

    axis_list = get_axis(selected_graph)

    #create a list for choose math functions
    math_functions = ["Sum", "Difference", "Mutliplication", "Division", "Average"]

    # Store the initial value of widgets 
    x_math_widget = True
    y_math_widget = True
        
    #create a drop down to choose axis
    x_axis_primary = st.sidebar.selectbox("Select X axis", axis_list, key="x_axis_primary")
    x_axis_secondary = st.sidebar.selectbox("Select X axis", [None] + axis_list, label_visibility="collapsed", key="x_axis_secondary")
    if x_axis_secondary is not None:
        x_math_widget = False
    else:
        x_math_widget = True
    x_axis_math = st.sidebar.selectbox("Select X axis", math_functions, label_visibility="collapsed", key="x_axis_math", disabled= x_math_widget)
    x_axis = axis_set(x_axis_primary, x_axis_secondary, x_axis_math)

    y_axis_primary = st.sidebar.selectbox("Select Y axis", axis_list, key="y_axis_primary")
    y_axis_secondary = st.sidebar.selectbox("Select Y axis", [None] + axis_list, label_visibility="collapsed", key="y_axis_secondary")
    if y_axis_secondary is not None:
        y_math_widget = False
    else:
        y_math_widget = True
    y_axis_math = st.sidebar.selectbox("Select Y axis", math_functions, label_visibility="collapsed", key="y_axis_math", disabled= y_math_widget)
    y_axis = axis_set(y_axis_primary, y_axis_secondary, y_axis_math)

    #change index to axis names
    #new_df.set_index('time', inplace=True)
    #set_df = new_df.loc[[x_axis, y_axis]]
    #st.write(set_df)

    st.write(get_data(all_df, selected_graph, x_axis_primary, y_axis_primary).transpose())
    
    #st.session_state[selected_graph] = update_dict(graph_title, x_axis, y_axis)
    
    st.write(st.session_state)
