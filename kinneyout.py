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
from functools import partial
import datetime

import plot
import csv

import os
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.elements.form import current_form_id

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd


#converting a list to a string for filenames
def listToString(s):
    # initialize an empty string
    str1 = ""
    for x in s:
        str1 = str1 + "__" + x
    return str1

#this checks if a dictionary has an element. 
def checkIfBlank(Dict,inputString):
    if (Dict[inputString] == ''):
        return False
    else :
        return True

def remove_duplicate_files(l):
    for d in l:
        if d['fn'] not in [d1['fn'] for d1 in st.session_state.file_list]:
            st.session_state.file_list.append(d)



#this the code to do math operations on arrays element wise for the standard plots axises. 
#current is a dictionary that contains the information about one standard plot
#StandardPlot_DF is a data frame that contains all of a vehicles plots
#Xdata1 is the name of the first column.
#xdata2 is the name of the second column that does the math. 
def mathStandardAxis(current,standardPlot_DF,XData1,XData2):
    #this extracts the XData1 data from standardPlot_DF and puts it in an np array. 
    standardXData1Values = np.array(standardPlot_DF.loc[current[XData1][3:]].values)
    
    #This checks if the XData2 function is not null. Otherwise it returns standardXData1Values
    if ( XData2 in current[XData2]):
        #this extracts the XData2 data from standardPlot_DF and puts it in an np array.
        standardXData2Values = np.array(standardPlot_DF.loc[current[XData2][3:]].values)        
        
        #this code does the actual operations on the arrays. 
        if(current['XOperation'] == "Sum"):
            return np.add(standardXData1Values, standardXData2Values) 
        
        elif(current['XOperation'] == "Difference"):
            return np.subtract(standardXData1Values, standardXData2Values) 
                    
        elif (current['XOperation'] == "Mutliplication"):
            return np.multiply(standardXData1Values, standardXData2Values)
                    
        elif (current['XOperation'] == "Division"):
            return np.divide(standardXData1Values, standardXData2Values)
                    
        elif (current['XOperation'] == "Average"):
            return_value1 = np.add(standardXData1Values, standardXData2Values) 
            return return_value1/2
                    
    else: 
        return standardXData1Values

#parse filename to get vehicle descriptor -- return a dict with names
def parse_filename(name):
    #removing .csv and splitting name based on '__'
    rem_dir = name.rsplit('/', 1)[-1]
    rem_type = rem_dir.split('.csv')
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
    if v is None:
        test_load = [ x[1] for x in d.index.tolist() ]
    else :
        vehicle = [ x for x in d.index.tolist() if x[0] == v ]
        test_load = [ x[1] for x in vehicle ]

    return test_load

def get_vehicle(d):
    return [ x[0] for x in d.index.tolist() ]

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
    else:
        return None

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
        st.session_state.load_case = [*set(get_load_case(s, None))]

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def linearization():
    with st.form(key="linearization_menu"):
        parameter = st.selectbox(label="linearization selectbox", options=st.session_state.graph_df_indices, 
                     index=0, key="linearization_data_select", label_visibility="collapsed")
        lower_bound = st.number_input("Min value: ", value=0.0, step=0.0001, 
                                  format='%.4f', key=f'linearization_lower_x')
        upper_bound = st.number_input("Max value: ", value=0.0, step=0.0001, 
                                  format='%.4f', key=f'linearization_upper_x')  
        
        submitted = st.form_submit_button("Enter")
        if submitted:
            idx = np.where(st.session_state.graph_df_indices==parameter)[0][0]
            X = np.array(st.session_state.new_graph_df.iloc[2*idx])
            Y = np.array(st.session_state.new_graph_df.iloc[2*idx+1])[(X>lower_bound) & (X<upper_bound)]
            X = X[(X>lower_bound) & (X<upper_bound)]
            
            if X.size == 0 or Y.size == 0: 
                st.error("Please select a range with data.", icon="🚨")
                print(f'X: {X}')
                print(f'Y: {Y}')
            else:
                return np.polyfit(X, Y, 1)[0]
            
#modifies new_graph_df with offsets from customization menu
def update_new_graph_df(x_offsets, y_offsets):
    for i, (x_off, y_off) in enumerate(zip(x_offsets, y_offsets)):
        st.session_state.new_graph_df.iloc[2*i] = st.session_state.graph_df.iloc[2*i] + x_off
        st.session_state.new_graph_df.iloc[2*i+1] = st.session_state.graph_df.iloc[2*i+1] + y_off

#copies graph_df into new_graph_df
def reset_new_graph_df():
    st.session_state.new_graph_df = st.session_state.graph_df.copy()

#creates widget/options to offset a given trace/line
def adjust_trace(label, idx, container):
    col1, col2 = container.columns(2)
    #only add tooltop to first row of widgets
    if idx == 0:
        x_off = col1.number_input(label + ' x-offset', value=0.0, step=0.0001, 
                                  format='%.4f', key=f'{label}{idx}_x-off',
                                  help="""Offset a line in the x-direction. Each line can have 
                                  its own x-offset. The widget which changes a given line is 
                                  labeled with the y-parameter used to create the line. If the 
                                  legends of the graph are unchanged, the y-parameters for the lines
                                  can be found there.""")
        y_off = col2.number_input(label + ' y-offset', value=0.0, step=0.0001, 
                                  format='%.4f', key=f'{label}{idx}_y-off',
                                  help="""Offset a line in the y-direction. Each line can have 
                                  its own y-offset. The widget which changes a given line is 
                                  labeled with the y-parameter used to create the line. If the 
                                  legends of the graph are unchanged, the y-parameters for the lines
                                  can be found there.""")
    else:
        x_off = col1.number_input(label + ' x-offset', value=0.0, step=0.0001, key=f'{label}{idx}_x-off')
        y_off = col2.number_input(label + ' y-offset', value=0.0, step=0.0001, key=f'{label}{idx}_y-off')
    return x_off, y_off

#creates region of the page with all the widgets to customize/adjust the plot
def customize_plot(fig):
    expander = st.expander('Adjust chart')
    with expander.form(key='update_plot'):
        #getting current axes ranges of the plot
        x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
        legends = []
        for trace in fig.data:
            x_mins = np.append(x_mins, np.min(trace['x']))
            x_maxs = np.append(x_maxs, np.max(trace['x']))
            y_mins = np.append(y_mins, np.min(trace['y']))
            y_maxs = np.append(y_maxs, np.max(trace['y']))
            legends = np.append(legends, trace['name'] if 'name' in trace else '')
        x_min = float(np.min(x_mins))
        x_max = float(np.max(x_maxs))
        y_min = float(np.min(y_mins))
        y_max = float(np.max(y_maxs))
        
        #options to adjust x-axis bounds
        col1_1, col1_2, col2_1, col2_2 = st.columns(4)
        x_axis_lower = col1_1.number_input('X-axis range', value=x_min, step=0.0001,
                                        format='%.4f', key='x_axis_lower',
                                        help="Enter a number to set as the lower bound of the x-axis of the graph.")
        x_axis_upper = col1_2.number_input(' ', value=x_max, step=0.0001, 
                                        format='%.4f', key='x_axis_upper', #label_visibility='hidden',
                                        help="Enter a number to set as the upper bound of the x-axis of the graph.")
        #options to adjust y-axis bounds
        y_axis_lower = col2_1.number_input('Y-axis range', value=y_min, step=0.0001, 
                                        format='%.4f', key='y_axis_lower',
                                        help="Enter a number to set as the lower bound of the y-axis of the graph.")
        y_axis_upper = col2_2.number_input(' ', value=y_max, step=0.0001, 
                                        format='%.4f', key='y_axis_upper', #label_visibility='hidden',
                                        help="Enter a number to set as the upper bound of the y-axis of the graph.")
        x_range = [x_axis_lower, x_axis_upper]
        y_range = [y_axis_lower, y_axis_upper]
        
        #options to adjust offsets of each trace/lines
        trace_update = np.array(list(map(partial(adjust_trace, container=st), legends, np.arange(len(legends)))))

        #options to add flags to the four quadrants
        col1, col2, col3, col4 = st.columns(4)
        quadrant1_show = col1.checkbox('Show Quadrant I flag', key='quadrant1_show')
        quadrant2_show = col2.checkbox('Show Quadrant II flag', key='quadrant2_show')
        quadrant3_show = col3.checkbox('Show Quadrant III flag', key='quadrant3_show')
        quadrant4_show = col4.checkbox('Show Quadrant IV flag', key='quadrant4_show')
        
        submitted = st.form_submit_button('Update chart')
        if expander.button('Reset chart', key='reset_chart_button'):
            reset_new_graph_df()
        if submitted:
            update_new_graph_df(trace_update[:,0].astype(float), trace_update[:,1].astype(float))
            new_fig = plot.update(fig, x_lim=x_range, y_lim=y_range,
                                quad1_show=quadrant1_show, quad2_show=quadrant2_show,
                                quad3_show=quadrant3_show, quad4_show=quadrant4_show,
                                x_offsets=trace_update[:,0].astype(float), y_offsets=trace_update[:,1].astype(float),
                                )
            return new_fig

    return fig


# initial page config
st.set_page_config(
     page_title="Kinney:Out",
     layout="wide",
     initial_sidebar_state="expanded",
)

# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

st.title("Welcome to the Kinney:Out Results Viewer")
#uploaded_files = st.file_uploader("Please select .csv files for data.", accept_multiple_files=True, type=['csv'])


#create a df that is a concationation of all .csv files
all_df = pd.DataFrame()

#initalize all session state variables
if 'graph_df' not in st.session_state:
    st.session_state.graph_df = pd.DataFrame()
if 'new_graph_df' not in st.session_state:
    st.session_state.new_graph_df = st.session_state.graph_df.copy()
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = pd.DataFrame()
if 'vehicle_list' not in st.session_state:
    st.session_state.vehicle_list = []
if 'load_case_list' not in st.session_state:
    st.session_state.load_case_list = []
if 'file_list' not in st.session_state:
    st.session_state.file_list = []

# Folder picker submit form
with st.form("file picker"):
    st.write('_(Optional)_ Input a starting path to browse files:')

    c1, c2 = st.columns([5,1])
    with c1:
        #get path by user
        file_path = st.text_input('Start path:', placeholder="C:\Documents\ ", label_visibility="collapsed")

        #copying path often puts in double quotes, however, path needs to to be without it
        #removes double quotes from give path if it exists
        try:
            file_path = file_path.lstrip('"')
        except:
            file_path = file_path
    with c2:
        #button to initiate browsing files
        browse_files_clicked = st.form_submit_button('Browse Files', help="User is able to choose a file path to start browsing files from.")

        if browse_files_clicked:
            fd_uploaded_files = fd.askopenfiles(master=root, initialdir=file_path, filetypes=[("CSV files","*.csv")])

            temp_all_df = pd.DataFrame()
            vehicle_list = []
            load_case_list = []
            filenames = []
            
            #go through all files and add it to a main dataframe
            for uploaded_file in fd_uploaded_files:
                #read csv file and convert to df
                df = pd.read_csv(uploaded_file)
                
                #parse the csv file name into a list of four segments
                file_strings = parse_filename(uploaded_file.name)

                #create a temporary df for manipulation
                temp_df = df

                #clean up the testbench name
                temp_df.iloc[:,0] = temp_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")

                #add another column based on file and add it to the index to create a MultiIndex
                temp_df = temp_df.assign(vehicle = file_strings["tv"])
                temp_df = temp_df.assign(test_load = file_strings["lc"])
                temp_df.rename({'time':'dataset'}, axis=1, inplace=True)
                temp_df.set_index(['vehicle','test_load','dataset'], inplace=True)

                #resets all column labels to be 0 to n rather than the time values
                temp_df.columns = range(temp_df.shape[1])

                #checking if user is adding a repeat dataset
                for d in st.session_state.file_list:
                    if d['df'].equals(temp_df):
                        c1.error('Duplicate file not added: ' + d['fn'], icon="🚨")
                        break
                else:
                    #add temp df to df holding all df's and files in session state
                    temp_all_df = pd.concat([temp_all_df, temp_df])

                    filename = (uploaded_file.name).rsplit('/', 1)[-1]
                    filenames.append({"fn":filename, "df":temp_df})
                continue
            
            #assigns variables to session state to be used even if the page reloads
            if not temp_all_df.empty:
                st.session_state.uploaded_df = pd.concat([st.session_state.uploaded_df, temp_all_df]).drop_duplicates(keep="first")
                remove_duplicate_files(filenames)
                c1.success('File(s) added: ' + str([d['fn'] for d in filenames]), icon="✅")

#file delete form
#DELETE LEAVE TWO SOMETIMES WHYYY
with st.form("delete files form", clear_on_submit=True):
    c1, c2 = st.columns([5,1])
    with c1:
        files = st.multiselect("delete files",  st.session_state.file_list, label_visibility="collapsed", format_func=lambda x: x['fn'])
    with c2:
        remove_files = st.form_submit_button('Remove Files', help="Remove files from the set that you are working with")

        if remove_files:
            for d in files:
                st.session_state.uploaded_df = pd.concat([st.session_state.uploaded_df, d['df']]).drop_duplicates(keep=False)
                st.session_state.file_list = [i for i in st.session_state.file_list if d['fn'] not in i['fn']]
            st.experimental_rerun()
        
            


#check whether user has uploaded any files
if len(st.session_state.uploaded_df) != 0:
    #if so, run through files and run rest of code
    dataset_list = []

    all_df = st.session_state.uploaded_df
    st.write(all_df)
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    all_axis_df = pd.concat([all_df, st.session_state.df])

    graph_tab, dataset_tab, standard_plot_tab = st.sidebar.tabs(["Graph", "Dataset Manipulation", "Standard Plot"])
    
    #TO DO: CHECK FOR MAX OF 5 GRAPHS

    with graph_tab:
        graph_selected_vehicle = st.selectbox("Select a vehicle", [*set(get_vehicle(all_axis_df))], key="graph_vehicle_select")
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
        selected_vehicle = st.selectbox("Select a vehicle", [*set(get_vehicle(all_axis_df))], key="vehicle_select")
        selected_lc = st.selectbox("Select a load case", [*set(get_load_case(all_axis_df, graph_selected_vehicle))], key="lc_select")
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
                try:
                    #adds to session state and creates row based on math functions
                    add_dataset(all_df, selected_vehicle, selected_lc, dataset_math, math_widget, rename_dataset)
                    #update all_axis_df based on added rows
                    all_axis_df = pd.concat([all_df, st.session_state.df])
                    st.success('New dataset created!', icon="✅")
                except:
                    st.error('Error: Dataset not created', icon="🚨")
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
        if 'standard_df' not in st.session_state:
            st.session_state.standard_df = pd.DataFrame()
        if 'standard_filename' not in st.session_state:
            st.session_state.standard_filename = ""
        with st.form('standard_plot_file'):
            #get path by user
            standard_file_path = st.text_input('Start path:', placeholder="C:\Documents\ ", label_visibility="collapsed")
            standard_file_clicked = st.form_submit_button('Add Standard Plot File', help="User is able to choose a file path to get standard plot file.")

            if standard_file_clicked:
                uploaded_file = fd.askopenfile(master=root, initialdir=standard_file_path, filetypes=[("CSV files","*.csv")])

                st.session_state.standard_filename = uploaded_file.name
                st.session_state.standard_df = pd.read_csv(uploaded_file)
        
        st.write("Chosen Standard Plot File: " + st.session_state.standard_filename)
        
        





    #make plot using user-selected rows of data. 
    if st.session_state.graph_df.empty == False:

        indices = np.array(st.session_state.graph_df.index)
        
        index_legends = []
        for tup in indices[np.arange(1, len(indices),2)]:
            index_legends = np.append(index_legends, tup[2])
            
        st.session_state["graph_df_indices"] = index_legends
            
        data_plot = plot.plot(st.session_state.graph_df, legends=index_legends, 
                              x_title=(indices[0])[2], y_title=(indices[1])[2],
                              title=(indices[1])[2]+' vs '+(indices[0])[2])

        new_data_plot = customize_plot(data_plot)
        
        #Plotly chart configurations
        config = dict({'scrollZoom': True,
                   'displayModeBar': True,
                   'editable': True})
        
        st.plotly_chart(new_data_plot, use_container_width=False, config=config)
        
        slope1 = linearization()
        if slope1:
            st.write("Slope: ", round(slope1, 4))
        st.write(st.session_state.graph_df)
        st.write(st.session_state.new_graph_df)

        graph_csv = convert_df(st.session_state.graph_df)
        col1, col2 = st.columns([3, 1])
        with col1:
            csv_name = st.text_input("Add custom name or leave blank", label_visibility="collapsed", placeholder="Add custom name or leave blank", key="csv_name")
            current_time = datetime.datetime.now()
            if not csv_name:
                csv_name = "graph" + str(current_time.year) +"-" + str(current_time.month) + "-" + str(current_time.day)
        with col2:
            st.download_button(label="Download data as CSV", data=graph_csv, file_name=csv_name + ".csv", mime='text/csv')


    #This is the code for the event handling when user presses the button labeled standard plots. 
    if st.button("Standard Plots"):
        # this is an array of dictionaries. Each dictionary will contain all
        # all the information about one of the standard plots. 
        standardPlots = [] 
    
        #Opens the standard plots csv file. Each row in this csv file contains information about one
        #standard plot. 
        with open('Kinney Standard Plots.csv','r') as csvFile: 
            csv_reader = csv.reader(csvFile) # csv in object form 
            
            # used to count what row the forloop is in.
            counter = 0 
            
            # iterates through every row in the csv file. Each row contains information about one standard plot
            for line in csv_reader:
                #this skips the first row in the csv file. The first row contains info about the 
                # characteristics. This is useful to look at for a person but not needed for the 
                # code
                if counter != 0:
                    #this is the actual dictionary that will get added to the standard plots array.
                    #this contains all the actual data for a single standard plot
                    standardPlot ={
                        "title" : line[0],
                        "xTitle" : line[1],
                        "yTitle" : line[2],
                        "DataFile": line[3],
                        "XData1" : line[4],
                        "XData2" : line[5],
                        "XOperation": line[6],
                        "XOffset": line[7],
                        "YData1" : line[8],
                        "YData2" : line[9],
                        "YOperation": line[10],
                        "YOffset": line[11],
                        "StandardLinearMin" : line[12],
                        "StandardLinearMax" : line[13],
                        "Quad1Flag" : line[14],
                        "Quad2Flag" : line[15],
                        "Quad3Flag" : line[16],
                        "Quad4Flag" : line[17],
                        
                    }
                    
                    #this adds the standard plot the standardPlots array 
                    standardPlots.append(standardPlot)

                # increments the line number
                counter = counter +1

        
       
        #go through all the standard plot axises. 
        for current in standardPlots:
            #goes through all the uploaded files.
            for file in uploaded_files:
                #if the standard plot matches the uploaded file it graphs data from that file. 
                if current['DataFile'] in file.name:
                    fileDict = parse_filename(file.name)

                    
                    #this gets the data from a data frame that contains all the files into
                    # into a data frame that only contains file that matches with current['datafile']
                    try:
                        # some of the vehicles files are named lateral while some are named fa__lateral
                        # this try block accounts for all of that 
                        standardPlot_DF = all_df.loc[fileDict['tv']].loc[current['DataFile']]
                    except: 
                        if current['DataFile'] == 'lateral':
                            standardPlot_DF = all_df.loc[fileDict['tv']].loc['fa__lateral']
                        
                    #these are the xvalues and yvalues put into a numpy array. See the 
                    #mathStandardAxis code for more details. 
                    xAxisValues = mathStandardAxis(current,standardPlot_DF,"XData1","XData2")
                    yAxisValues = mathStandardAxis(current,standardPlot_DF,"YData1","YData2")

                    #these are the x and yaxises. They will take all the values in xAxisValues and yAxisValues
                    #and account for the linear min and max 
                    trimmedXAxis = []
                    trimmedYAxis = []

                    #this is the standard linear max and min. It starts out being the yaxise's min and max
                    #values. 
                    stdLinMax = max(yAxisValues)
                    stdLinMin = min(yAxisValues)

                    #if the plot has a standard linear max or min the values above get changed to account for 
                    # that. Otherwise trimmed x and y axis just become copies of xAxisValues and yAxisValues. 
                    if (current["StandardLinearMin"] != ''):
                        stdLinMin = current['StandardLinearMin']
                    
                    if (current["StandardLinearMax"] != ''):
                        stdLinMax = current['StandardLinearMax']

                    #this is the data plot that gets graphed 
                    data_plot = plot.plot(
                        [xAxisValues,yAxisValues],
                        title = current['title'],
                        y_title = current['yTitle'],
                        x_title = current['xTitle'],
                        quad1_title = current['Quad1Flag'],
                        quad2_title = current['Quad2Flag'],
                        quad3_title = current['Quad3Flag'],
                        quad4_title = current['Quad4Flag']                 
                    )

                    #this sets the x and y offests 
                    x_offsetValue = 0
                    y_offsetValue = 0
                    if checkIfBlank(current,'XOffset'):
                        x_offsetValue = current['XOffset']
                    if checkIfBlank(current,'YOffset'):
                        y_offsetValue = current['YOffset']
                

                    #this sets the offests,quadrant flags, and x and ytitles. 
                    data_plot = plot.update(data_plot,
                        x_offsets = [x_offsetValue],
                        y_offsets = [y_offsetValue]
                    )


                    #st.write(stdLinMax)
                    #st.write(stdLinMin)
                    #this graphs the plot. 
                    config = dict({'scrollZoom': True,
                        'displayModeBar': True,
                        'editable': True})
                    st.plotly_chart(data_plot, config=config)

