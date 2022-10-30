# importing the required module
from codecs import ignore_errors
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

numYaxises = 0


#This function preforms math functions on two lists element wise
#It returns a numpy list 
#param set_df: The data frame that contains the axises we are extracting data from
#param axis: the first list that has math done on it. This is the left side of the operation
#param axis_secondary: The other list that gets math done on it. This is the right side of the operation
#param axis_math: This specifies which opera tation to do. 
def mathTwoLists(set_df,axis,axis_secondary,axis_math):
    axis1 = np.array(set_df.loc[axis])
    axis2 = np.array(set_df.loc[axis_secondary])
    returnValue = []
    print(axis_math)
    if(axis_math == "Sum"):
        returnValue = np.add(axis1, axis2) 
    elif(axis_math == "Difference"):
        returnValue = np.subtract(axis1, axis2) 
    elif (axis_math == "Mutliplication"):
        returnValue = np.multiply(axis1, axis2)
    elif (axis_math == "Division"):
        returnValue = np.divide(axis1, axis2)
    elif (axis_math == "Average"):
        returnValue1 = np.add(axis1, axis2) 
        returnValue = returnValue1/2
    return returnValue

def math(axis_list, operation):
    returnValue = []
    for cord in axis_list[0]:
        if operation == "Sum":
            returnValue.append(0) 
        if operation == "Difference" or operation == "Division":
            returnValue.append(cord)
        if operation == "Mutliplication":
            returnValue.append(1) 
        
    if operation == "Sum":
        for list in axis_list:
            counter = 0
            for cord in list:
                returnValue[counter] = returnValue[counter] + cord
                counter = counter + 1
    if operation == "Mutliplication":
        for list in axis_list:
            counter = 0
            for cord in list:
                returnValue[counter] = returnValue[counter] * cord
                counter = counter + 1
    if operation == "Difference": 
        counter = 0
        for list in axis_list:
            counter2 = 0
            if counter > 0:
                for cord in list:
                    returnValue[counter2] = returnValue[counter2] - cord
                    counter2 = counter2 + 1    
            counter = counter + 1
    
    if operation == "Division": 
        counter = 0
        for list in axis_list:
            counter2 = 0
            if counter > 0:
                for cord in list:
                    returnValue[counter2] = returnValue[counter2] / cord
                    counter2 = counter2 + 1    
            counter = counter + 1 
    return returnValue

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
    axis_list = []
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
    axis_list = get_axis(selected_graph)
    
    #create a list for choose math functions
    math_functions = ["Sum", "Difference", "Mutliplication", "Division", "Average"]

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

        #change index to axis names
        #if cancel:
            #all_df.set_index('time', inplace=True)
        
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
        

        
  

    #plot_it("x_axis_key","x_axis_secondary_key","x_axis_math_key","y_axis_key","y_axis_secondary_key","y_axis_math_key",True)
    #plot_it("x_axis_key2","x_axis_secondary_key2","x_axis_math_key2","y_axis_key2","y_axis_secondary_key2","y_axis_math_key2",False)
    #plot_it("x_axis_key3","x_axis_secondary_key3","x_axis_math_key3","y_axis_key3","y_axis_secondary_key3","y_axis_math_key3",False)
    #plot_it("x_axis_key4","x_axis_secondary_key4","x_axis_math_key4","y_axis_key4","y_axis_secondary_key4","y_axis_math_key4",False)
    #plot_it("x_axis_key5","x_axis_secondary_key5","x_axis_math_key5","y_axis_key5","y_axis_secondary_key5","y_axis_math_key5",False)
    st.write(all_df)
    yAxises = []
    #xAxis = plot_itV2("x_axis_key", "x_axis_secondary_key", "x_axis_math_key", True,"Select the X axis",False)
    #yAxis1 = plot_itV2("y_axis_key", "y_axis_secondary_key", "y_axis_math_key", False,"Select the Y axis",False)
    #yAxis2 = plot_itV2("y_axis_key2", "y_axis_secondary_key2", "y_axis_math_key2", False,"OPTIONAL: Select the another Y axis to graph",True)
    #yAxis3 = plot_itV2("y_axis_key3", "y_axis_secondary_key3", "y_axis_math_key3", False,"OPTIONAL: Select the another Y axis to graph",True)
    #yAxis4 = plot_itV2("y_axis_key4", "y_axis_secondary_key4", "y_axis_math_key4", False,"OPTIONAL: Select the another Y axis to graph",True)
    #yAxis5 = plot_itV2("y_axis_key5", "y_axis_secondary_key5", "y_axis_math_key5", False,"OPTIONAL: Select the another Y axis to graph",True)
    #print(yAxis2)
    
    
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

    #get data
    set_df = get_data(all_df, selected_graph, x_axis_primary, y_axis_primary).transpose()
    st.write(set_df)

    #make plot using user-selected rows of data
    data_plot = plot.plot(set_df)
    st.plotly_chart(data_plot)
