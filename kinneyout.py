# importing the required module
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st


import plot # gc

# initial page config
st.set_page_config(
     page_title="Kinney:Out",
     layout="wide",
     initial_sidebar_state="expanded",
)

#parse filename to get vehicle descriptors
#return a dict with names
def parse_filename(name):
    #removing .csv and splitting name based on '__'
    rem_type = name.split('.csv')
    str_name = rem_type[0].split('__')

    #concatting the seperated strings to give the correct vehicle descriptors
    model_template = str(str_name[0]) + "__" + str(str_name[1]) + "__" + str(str_name[2])
    test_variant = str(str_name[3]) + "__" + str(str_name[4]) + "__" + str(str_name[5])
    test_bench = str(str_name[6]) + "__" + str(str_name[7])
    test_load_case = str(str_name[8])

    return {"mt": model_template, "tv": test_variant, "tb": test_bench, "tlc": test_load_case}

#remove duplicates in a list
def rem_duplicates(l):
    new_list = [*set(l)]
    return new_list

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
uploaded_files = st.file_uploader("Please select up to 5 datasets to be graphed.", accept_multiple_files=True)

#create a df to hold concatanation of all datasets
full_df = pd.DataFrame()

#check if user has uploaded more than 5 files
if len(uploaded_files) > 5: 
    st.write("Error: Too many files. Please only select up to 5 datasets")

#check whether user has uploaded any files
if len(uploaded_files) != 0 and len(uploaded_files) <= 5:
    #if so, run through files and run rest of code
    axis_list = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        
        filename = parse_filename(uploaded_file.name)
        st.write("filename:", filename["mt"])
        st.write(df)
       

        #create a new df for manipulating
        new_df = df
        new_df.iloc[:,0] = new_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")
        #st.write(new_df)
        full_df = pd.concat([full_df, new_df], ignore_index= True)

        #st.write("full_df")
        #st.write(full_df[1])
        #st.write("full_df")
        
        # create bar graph to display data
        #fig, ax = plt.subplots()

        #ax.plot(df.iloc[0].head(5), df.iloc[1].head(5))

        #st.pyplot(fig)

    #st.write(full_df)

    #stop here stop here stop here stop here stop here stop here stop here stop heer

        #remove duplicates and sort list alphabetically
        axis_list = sorted(rem_duplicates(full_df.iloc[:,0]))

        #remove duplicates and sort list alphabetically
        sortedlist = sorted(rem_duplicates(full_df.iloc[:,0]))
        for value in sortedlist:
            axis_list.append(value)
    
    #create a list for choose math functions
    math_functions = [None, "Sum", "Difference", "Mutliplication", "Division", "Average"]

    def plot_it(x_axis_key,x_axis_secondary_key,x_axis_math_key,y_axis_key,y_axis_secondary_key,y_axis_math_key, cancel):

        axises_array = [] #all the arrays that get plotted

    
        # Store the initial value of widgets 
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

        #change index to axis names
        if cancel:
            full_df.set_index('time', inplace=True)
        
        st.write(full_df)
   
        #set_df = full_df.loc[[x_axis, y_axis]]
        set_df = full_df.loc[axises_array]
        st.write(set_df)

        mathXAxis = []
        mathYAxis = []  

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
    
    plot_it("x_axis_key","x_axis_secondary_key","x_axis_math_key","y_axis_key","y_axis_secondary_key","y_axis_math_key",True)
    plot_it("x_axis_key2","x_axis_secondary_key2","x_axis_math_key2","y_axis_key2","y_axis_secondary_key2","y_axis_math_key2",False)
    plot_it("x_axis_key3","x_axis_secondary_key3","x_axis_math_key3","y_axis_key3","y_axis_secondary_key3","y_axis_math_key3",False)
    plot_it("x_axis_key4","x_axis_secondary_key4","x_axis_math_key4","y_axis_key4","y_axis_secondary_key4","y_axis_math_key4",False)
    plot_it("x_axis_key5","x_axis_secondary_key5","x_axis_math_key5","y_axis_key5","y_axis_secondary_key5","y_axis_math_key5",False)