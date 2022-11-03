# importing the required module
from cgitb import reset
from codecs import ignore_errors
import re
from select import select
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from requests import session
import streamlit as st

import plot



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

#this returns a numpy array of two lists added together 
def math(df, tb, x, y, axis_math):
    axis1 = np.array(df.loc[tb].loc[x].values)#gets the x-axis from the df
    axis2 = np.array(df.loc[tb].loc[y].values)#gets the other x-axis from the df
    
    return_value = []
    if(axis_math == "Sum"):
        return_value = np.add(axis1, axis2) 
    elif(axis_math == "Difference"):
        return_value = np.subtract(axis1, axis2) 
    elif (axis_math == "Mutliplication"):
        return_value = np.multiply(axis1, axis2)
    elif (axis_math == "Division"):
        return_value = np.divide(axis1, axis2)
    elif (axis_math == "Average"):
        return_value1 = np.add(axis1, axis2) 
        return_value = return_value1/2
    print(return_value)
    return return_value

#get second index based on df and keyword passed in paramenter, return list
def get_axis(d, key):
    temp_list = [ x for x in d.index.tolist() if x[0] == key ]
    axis = [x[1] for x in temp_list]
    return axis

#create a new axis based on axises selected and math function
def create_axis(df, ds, a_math, m):
    new_axis = None
    #error checking to make sure only two selected
    if len(a_math) == 2:
        #create new axis
        new_axis = math(df, ds, a_math[0], a_math[1], m)
        st.success('New axis added', icon="✅")
    else:
        st.error('Please select two axises to create a new axis', icon="🚨")

    #return array of values
    return new_axis

#add axis to session state df (the df which we are holding everything)
def add_axis(df, ds, a_math, m):
    #get new values
    new_vals = create_axis(df, ds, a_math, m)

    #making sure two values were added
    if new_vals is not None:
        #ensure df is correct shape
        temp_df = pd.DataFrame(new_vals).transpose()
        #add new indices to match main df
        new_df = temp_df.assign(test=ds) 
        new_df = new_df.assign(dataset=m + "(" + a_math[0] + ", " + a_math[1] + ")")
        new_df.set_index(['test','dataset'], inplace=True)
        #update session state to call new values
        st.session_state.df = pd.concat([st.session_state.df, new_df])

#create dataframe for graphing based on selected axises
def create_graph_df(d, x, y):
    x_df = d.query('test == @selected_graph and dataset == @x')
    y_df = d.query('test == @selected_graph and dataset == @y')
    return pd.concat([x_df, y_df])

    

st.title("Welcome to the Kinney:Out Results Viewer")
uploaded_files = st.file_uploader("Please select datasets to be graphed.", accept_multiple_files=True, type=['csv'])

#create a df that is a concationation of all .csv files
all_df = pd.DataFrame()

if 'graph_df' not in st.session_state:
        st.session_state.graph_df = pd.DataFrame()

#create a list to hold all datasets
df_list = []
df_name_list = []

#check whether user has uploaded any files
if len(uploaded_files) != 0:
    #if so, run through files and run rest of code
    axis_list = []

    #go through all files and add it to a main dataframe
    for uploaded_file in uploaded_files:
        #read csv file and convert to df
        df = pd.read_csv(uploaded_file)

        #parse the csv file name into a list of four segments
        filenames = parse_filename(uploaded_file.name)

        #add filename to list to choose from
        df_name_list.append(filenames['tv']+filenames["tlc"])

        #create a temporary df for manipulation
        temp_df = df

        #clean up the testbench name
        temp_df.iloc[:,0] = temp_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")

        #add another column based on file and add it to the index to create a MultiIndex
        temp_df = temp_df.assign(test = filenames["tv"]+filenames["tlc"])
        temp_df.rename({'time':'dataset'}, axis=1, inplace=True)
        temp_df.set_index(['test','dataset'], inplace=True)

        #add temp df to df holding all df's
        all_df = pd.concat([all_df, temp_df])

    #resets all column labels to be 0 to n rather than the time values
    all_df.columns = range(all_df.shape[1])

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
        
    all_axis_df = pd.concat([all_df, st.session_state.df])
    
    #create a list for choose math functions
    math_functions = ["Sum", "Difference", "Mutliplication", "Division", "Average"]

    #select box widget to choose vehicle dataset
    selected_graph = st.sidebar.selectbox("Select graph to work on", df_name_list, key="data_select")
    axis_list = get_axis(all_df, selected_graph)

    #form to create a new axis based on math functions
    with st.sidebar.form("add axis", clear_on_submit=True):
        #multiselect widget to select two axises
        axis_math = st.multiselect("Select two existing axises to create a new axis", axis_list, key="axis_math", max_selections=2)
        #select widget to select a math function
        math_widget = st.selectbox("Select math function", math_functions, label_visibility="collapsed", key="math_widget")

        #form sumbit button
        add_axis_submit = st.form_submit_button("Create new axis")
        if add_axis_submit:
            #adds to session state and creates row based on math functions
            add_axis(all_df, selected_graph, axis_math, math_widget)
            #update all_axis_df based on added rows
            all_axis_df = pd.concat([all_df, st.session_state.df])
    
    #form to create graph
    with st.sidebar.form("add graph", clear_on_submit=True):
        x_axis = st.selectbox("Select an X axis", axis_list, key="x_axis")
        y_axis = st.multiselect("Select up to five Y axises", get_axis(all_axis_df, selected_graph), key="y_axis", max_selections=5)
        update_graph_submit = st.form_submit_button("Update graph")
        if update_graph_submit:
            st.session_state.graph_df = create_graph_df(all_axis_df, x_axis, y_axis)
            print(st.session_state.graph_df)

    #form to delete graph
    with st.sidebar.form("del axis", clear_on_submit=True):
        #multiselect widget to pull ONLY created axis
        sel_axis = st.multiselect("Select an axis to delete", get_axis(st.session_state.df, selected_graph), key="del_axis")
        #create a temp df to work on
        temp_session_state = st.session_state.df

        #form submit button
        del_axis_submit = st.form_submit_button("Delete axis")
        if del_axis_submit:
            #interate through selected axises
            for x in sel_axis:
                #delete row based on selected axises
                temp_session_state.drop(index=(selected_graph, x), inplace=True)
            #update session state based on deleted rows
            st.session_state.df = temp_session_state
            #update all_axis_df based on deleted rows
            all_axis_df = pd.concat([all_df, st.session_state.df])


    st.write(st.session_state.df)
    st.write(all_axis_df)  
    st.write(st.session_state.graph_df)    

    #make plot using user-selected rows of data. 
    if st.session_state.graph_df.empty == False:
        data_plot = plot.plot(st.session_state.graph_df)
        st.plotly_chart(data_plot)
