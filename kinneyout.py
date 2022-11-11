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
import datetime
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

    return {"mt": model_template, "tv": test_variant, "tb": test_bench, "lc": test_load_case}

#this returns a numpy array of two lists added together 
def math(df, v, lc, x, y, dataset_math):
    x_row = df.query('vehicle == @v and test_load == @lc and dataset == @x')
    y_row = df.query('vehicle == @v and test_load == @lc and dataset == @y')
    axis1 = np.array(x_row.values) #gets the x-axis from the df
    axis2 = np.array(y_row.values) #gets the other x-axis from the df
    
    return_value = []
    if(dataset_math == "Sum"):
        return_value = np.add(axis1, axis2) 
    elif(dataset_math == "Difference"):
        return_value = np.subtract(axis1, axis2) 
    elif (dataset_math == "Mutliplication"):
        return_value = np.multiply(axis1, axis2)
    elif (dataset_math == "Division"):
        return_value = np.divide(axis1, axis2)
    elif (dataset_math == "Average"):
        return_value1 = np.add(axis1, axis2) 
        return_value = return_value1/2
    return return_value

#get second index based on df and keyword passed in paramenter, return list
def get_load_case(d, v):
    vehicle = [ x for x in d.index.tolist() if x[0] == v ]
    test_load = [ x[1] for x in vehicle ]

    return test_load

#get second index based on df and keyword passed in paramenter, return list
def get_dataset(d, v, tl):
    vehicle = [ x for x in d.index.tolist() if x[0] == v ]
    test_load = [ x for x in vehicle if x[1] == tl ]
    dataset = [ x[2] for x in test_load ]

    return dataset

#create a new axis based on axes selected and math function
def create_axis(df, v, tl, a_math, m):
    new_axis_vals = None
    #error checking to make sure only two selected
    if len(a_math) == 2:
        #create new axis
        new_axis_vals = math(df, v, tl, a_math[0], a_math[1], m)
        st.success('New dataset created', icon="✅")
    else:
        st.error('Please select two datasets to create a new dataset', icon="🚨")

    #return array of values
    return new_axis_vals

#add dataset to session state df (the df which we are holding everything)
def add_dataset(df, v, tl, a_math, m, name):
    #get new values
    new_vals = create_axis(df, v, tl, a_math, m)
    #checking if user added custom name, if not use default
    if not name:
        name = m + "(" + a_math[0] + ", " + a_math[1] + ")"

    #making sure two values were added
    if new_vals is not None:
        #ensure df is correct shape
        temp_df = pd.DataFrame(new_vals)
        #add new indices to match main df
        new_df = temp_df.assign(vehicle=v) 
        new_df = new_df.assign(test_load=tl) 
        new_df = new_df.assign(dataset=name)
        new_df.set_index(['vehicle','test_load','dataset'], inplace=True)
        #update session state to call new values
        st.session_state.df = pd.concat([st.session_state.df, new_df])

#create dataframe for graphing based on selected axes
def create_graph_df(d, v, lc, x, y):
    x_df = d.query('vehicle == @v and test_load == @lc and dataset == @x')
    y_df = d.query('vehicle == @v and test_load == @lc and dataset == @y')
    return_df = pd.concat([x_df, y_df])

    return pd.concat([st.session_state.graph_df, return_df])


def check_graph_lc(aad, s, gcv):
    if st.session_state.graph_df.empty:
        st.session_state.load_case = [*set(get_load_case(aad, gcv))]
    else:
        st.session_state.load_case = [*set(get_load_case(s, gcv))]

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

    

st.title("Welcome to the Kinney:Out Results Viewer")
uploaded_files = st.file_uploader("Please select .csv files for data.", accept_multiple_files=True, type=['csv'])

#create a df that is a concationation of all .csv files
all_df = pd.DataFrame()

if 'graph_df' not in st.session_state:
        st.session_state.graph_df = pd.DataFrame()

#create a list to hold all datasets
vehicle_list = []
load_case_list = []

#check whether user has uploaded any files
if len(uploaded_files) != 0:
    #if so, run through files and run rest of code
    dataset_list = []

    #go through all files and add it to a main dataframe
    for uploaded_file in uploaded_files:
        #read csv file and convert to df
        df = pd.read_csv(uploaded_file)

        #parse the csv file name into a list of four segments
        filenames = parse_filename(uploaded_file.name)

        #add filename to list to choose from
        vehicle_list.append(filenames['tv'])
        load_case_list.append(filenames["lc"])

        #create a temporary df for manipulation
        temp_df = df

        #clean up the testbench name
        temp_df.iloc[:,0] = temp_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")

        #add another column based on file and add it to the index to create a MultiIndex
        temp_df = temp_df.assign(vehicle = filenames["tv"])
        temp_df = temp_df.assign(test_load = filenames["lc"])
        temp_df.rename({'time':'dataset'}, axis=1, inplace=True)
        temp_df.set_index(['vehicle','test_load','dataset'], inplace=True)

        #add temp df to df holding all df's
        all_df = pd.concat([all_df, temp_df])

    #resets all column labels to be 0 to n rather than the time values
    all_df.columns = range(all_df.shape[1])

    #remove duplicated from the each list
    vehicle_list = [*set(vehicle_list)]
    load_case_list = [*set(load_case_list)]

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
        
    all_axis_df = pd.concat([all_df, st.session_state.df])

    graph_tab, dataset_tab, standard_plot_tab = st.sidebar.tabs(["Graph", "Dataset Manipulation", "Standard Plot"])
    with graph_tab:
        graph_selected_vehicle = st.selectbox("Select a vehicle", vehicle_list, key="graph_vehicle_select")
        if 'load_case' not in st.session_state:
            st.session_state.load_case = [*set(get_load_case(all_axis_df, graph_selected_vehicle))]
        check_graph_lc(all_axis_df, st.session_state.graph_df, graph_selected_vehicle)
        graph_selected_lc = st.selectbox("Select a load case", st.session_state.load_case, key="graph_selected_lc")
        x_axis = st.selectbox("Select an X axis", get_dataset(all_axis_df, graph_selected_vehicle, graph_selected_lc), key="x_axis")
        y_axis = st.selectbox("Select a Y axis", get_dataset(all_axis_df, graph_selected_vehicle, graph_selected_lc), key="y_axis")

        col1, col2, col3 = st.columns([12,16,12], gap='small')
        with col1:
            update_graph = st.button("Add graph",  key="add graph")
            if update_graph:
                st.session_state.graph_df = create_graph_df(all_axis_df, graph_selected_vehicle, graph_selected_lc, x_axis, y_axis)
                st.experimental_rerun()
        with col2:
            del_graph = st.button("Delete last graph", key="del graph")
            if del_graph:
                st.session_state.graph_df = st.session_state.graph_df[:-2]
                st.experimental_rerun()
        with col3:
            del_all_graph = st.button("Clear graph", key="clear graph")
            if del_all_graph:
                st.session_state.graph_df = pd.DataFrame()
                st.experimental_rerun()

    with dataset_tab:
        #select box widget to choose vehicle dataset
        selected_vehicle = st.selectbox("Select a vehicle", vehicle_list, key="vehicle_select")
        selected_lc = st.selectbox("Select a load case", load_case_list, key="lc_select")
        dataset_list = get_dataset(all_df, selected_vehicle, selected_lc)

        #create a list for choose math functions
        math_functions = ["Sum", "Difference", "Mutliplication", "Division", "Average"]

        #form to create a new axis based on math functions
        with st.form("add dataset", clear_on_submit=True):
            #multiselect widget to select two axes
            dataset_math = st.multiselect("Select two existing datasets to create a new dataset", dataset_list, key="dataset_math", max_selections=2)
            #select widget to select a math function
            math_widget = st.selectbox("Select math function", math_functions, label_visibility="collapsed", key="math_widget")
            rename_dataset = st.text_input("Add custom name or leave blank", label_visibility="collapsed", placeholder="Add custom name or leave blank", key="rename_dataset")

            #form sumbit button
            add_ds_submit = st.form_submit_button("Create new dataset")
            if add_ds_submit:
                #adds to session state and creates row based on math functions
                add_dataset(all_df, selected_vehicle, selected_lc, dataset_math, math_widget, rename_dataset)
                #update all_axis_df based on added rows
                all_axis_df = pd.concat([all_df, st.session_state.df])
                st.experimental_rerun()

        #form to delete created dataset
        with st.form("del dataset", clear_on_submit=True):
            #multiselect widget to pull ONLY created axis
            sel_ds = st.multiselect("Select a dataset to delete", get_dataset(st.session_state.df, selected_vehicle, selected_lc), key="del_dataset")
            #create a temp df to work on
            temp_session_state = st.session_state.df

            #form submit button
            del_ds_submit = st.form_submit_button("Delete dataset")
            if del_ds_submit:
                #interate through selected datasets
                for x in sel_ds:
                    #delete row based on selected axes
                    temp_session_state.drop(index=(selected_vehicle, selected_lc, x), inplace=True)
                #update session state based on deleted rows
                st.session_state.df = temp_session_state
                #update all_axis_df based on deleted rows
                all_axis_df = pd.concat([all_df, st.session_state.df])
                st.experimental_rerun()
    with standard_plot_tab:
        st.write("template")
    #make plot using user-selected rows of data. 
    if st.session_state.graph_df.empty == False:
        data_plot = plot.plot(st.session_state.graph_df, title="Bump Steer: Vehicle A, B, and C", x_title="wheel_yaw_angle_left", y_title="Vehicle A, B, C", legends=(["wheel_travel", "wheel_roll_angle", "wheel_yaw_angle"]))
        st.plotly_chart(data_plot, use_container_width=True)
        st.write(st.session_state.graph_df) 
    
    #st.write(st.session_state.df)
    #st.write(all_axis_df)  
       

        graph_csv = convert_df(st.session_state.graph_df)
        col1, col2 = st.columns([3, 1])
        with col1:
            csv_name = st.text_input("Add custom name or leave blank", label_visibility="collapsed", placeholder="Add custom name or leave blank", key="csv_name")
            current_time = datetime.datetime.now()
            if not csv_name:
                csv_name = "graph" + str(current_time.year) +"-" + str(current_time.month) + "-" + str(current_time.day)
        with col2:
            st.download_button(label="Download data as CSV", data=graph_csv, file_name=csv_name + ".csv", mime='text/csv')

