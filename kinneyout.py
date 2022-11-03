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

#get second index based on df and keyword passed in paramenter, return list
def get_axis(d, key):
    temp_list = [ x for x in d.index.tolist() if x[0] == key ]
    axis = [x[1] for x in temp_list]
    return axis

#create dict based on graph selection and options
def update_dict(df_name, x, y):
    return {"df_name":df_name, "x_axis":x[0], "x_secondary":x[1], "x_math":x[2], "y_axis":y[0], "y_secondary":y[1], "y_math":y[2]}

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

#update session state
def update_session_state(df, ds, a_math, m):
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
    

st.title("Welcome to the Kinney:Out Results Viewer")
uploaded_files = st.file_uploader("Please select datasets to be graphed.", accept_multiple_files=True, type=['csv'])

#create a df that is a concationation of all .csv files
all_df = pd.DataFrame()

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
        temp_df.set_index(['test','time'], inplace=True)
        temp_df.rename(index={'time':'dataset'})

        #add tempoary df to df holding all df's
        all_df = pd.concat([all_df, temp_df])

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
            update_session_state(all_df, selected_graph, axis_math, math_widget)
            #update all_axis_df based on added rows
            all_axis_df = pd.concat([all_df, st.session_state.df])
    
    #form to create graph
    with st.sidebar.form("add graph"):
        x_axis = st.selectbox("Select an X axis", axis_list, key="x_axis")
        y_axis = st.multiselect("Select up to five Y axises", get_axis(all_axis_df, selected_graph), key="y_axis")
        update_graph_submit = st.form_submit_button("Update graph")
        if update_graph_submit:
            st.write("axises", x_axis, y_axis)

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
    

    def plot_it(x_axis_key,x_axis_secondary_key,x_axis_math_key,y_axis_key,y_axis_secondary_key,y_axis_math_key, cancel): 

        axises_array = [] #all the arrays that get plotted

    
        # Store the initial math value of widgets 
        x_math_widget = True
        y_math_widget = True
        
        #create a drop down to choose axis
        x_axis = st.sidebar.selectbox("Select X axis", axis_list, key=x_axis_key)
        axises_array.append(x_axis)
    

        x_axis_secondary = st.sidebar.selectbox("Select X axis", [None] + axis_list, label_visibility="collapsed", key= x_axis_secondary_key)
        if x_axis_secondary is not None:
            x_math_widget = False
            axises_array.append(x_axis_secondary)
        else:
            x_math_widget = True
        x_axis_math = st.sidebar.selectbox("Select X axis", math_functions, label_visibility="collapsed", key=x_axis_math_key, disabled= x_math_widget)

        y_axis = st.sidebar.selectbox("Select Y axis", axis_list, key=y_axis_key)
        axises_array.append(y_axis)
    
        y_axis_secondary = st.sidebar.selectbox("Select Y axis", [None] + axis_list, label_visibility="collapsed", key=y_axis_secondary_key)
        if y_axis_secondary is not None:
            y_math_widget = False
            axises_array.append(y_axis_secondary)
        
        else:
            y_math_widget = True
        y_axis_math = st.sidebar.selectbox("Select Y axis", math_functions, label_visibility="collapsed", key=y_axis_math_key, disabled= y_math_widget)
        
        if st.sidebar.button("Add an aditional graph"):
            print("The user wants to make an additional graph")
        
        st.write(all_df)
   
        #set_df = all_df.loc[[x_axis, y_axis]]
        set_df = all_df.loc[axises_array]
        st.write(set_df)

        mathYAxis = [] 
        mathXAxis = []

        if not x_math_widget:
            mathXAxis = mathTwoLists(set_df,x_axis,x_axis_secondary,x_axis_math)
        else:
            if x_axis == y_axis:
                helperx = np.array(set_df.loc[x_axis])
                mathXAxis = helperx[0]
            else:
                mathXAxis = np.array(set_df.loc[x_axis])
            

        if not y_math_widget:
            mathYAxis = mathTwoLists(set_df,y_axis,y_axis_secondary,y_axis_math) 
        else:
            if x_axis == y_axis:
                helpery = np.array(set_df.loc[y_axis])
                mathYAxis = helpery[0]
            else:
                mathYAxis = np.array(set_df.loc[y_axis])


        

        #make plot using user-selected rows of data
        data_plot = plot.plot_plotly(np.array([mathXAxis, mathYAxis]))
        st.plotly_chart(data_plot)
    def plot_itV2(axis_key, axis_secondary_key,axis_math_key,cancel,sideBarName,optional):
        axises_array = [] #all the arrays that get plotted

        # Store the initial math value of widgets 
        math_widget = True
        
        #create a drop down to choose axis
        if not optional:
            axis = st.sidebar.selectbox(sideBarName, axis_list, key=axis_key)
        else:
            axis = st.sidebar.selectbox(sideBarName, [None] + axis_list, key=axis_key)
        
        axises_array.append(axis)


        axis_secondary = st.sidebar.selectbox(sideBarName, [None] + axis_list, label_visibility="collapsed", key= axis_secondary_key)
        if axis_secondary is not None:
            math_widget = False
            axises_array.append(axis_secondary)
        else:
            math_widget = True
        axis_math = st.sidebar.selectbox(sideBarName, math_functions, label_visibility="collapsed", key=axis_math_key, disabled= math_widget)

        

        #change index to axis names
        print(all_df)
        if cancel:
            all_df.set_index('time', inplace=True)
        
        set_df = all_df.loc[axises_array]
        
        if axis == "None":
            arr = np.empty(0)
            return arr

        mathAxis = []
        if not math_widget:
            mathAxis = mathTwoLists(set_df,axis,axis_secondary,axis_math)
        else:
            mathAxis = np.array(set_df.loc[axis])
        
        return mathAxis
        
   
    #These Axises get graphed
    mathYAxis = [] 
    mathXAxis = []

    #Adds the two arrays together element wise if the user has decided to preform math operations.
    #This new array becomes the axis. Otherwise it just uses the singular graph 
    #if x_math_widget == False:
    #    mathXAxis = mathV3(all_df, selected_graph, x_axis_primary, x_axis_secondary,x_axis_math)
    #else: 
    #    mathXAxis = np.array(all_df.loc[selected_graph].loc[x_axis_primary].values)

    #Adds the two arrays together element wise if the user has decided to preform math operations.
    #This new array becomes the axis. Otherwise it just uses the singular graph       
    #if y_math_widget == False:
    #   mathYAxis = mathV3(all_df, selected_graph, y_axis_primary, y_axis_secondary,y_axis_math)
    #else:
    #    mathYAxis = np.array(all_df.loc[selected_graph].loc[y_axis_primary].values)
    #mathXAxis = np.array(all_axis_df.loc[selected_graph].loc[a].values)
    #mathYAxis = np.array(all_axis_df.loc[selected_graph].loc[y_axis_primary].values)
    print(mathYAxis)
    print(mathXAxis)

    #make plot using user-selected rows of data. 
    data_plot = plot.plot(np.array([mathXAxis, mathYAxis]))
    st.plotly_chart(data_plot)
