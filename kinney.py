# importing the required module
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

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

uploaded_files = st.file_uploader("Choose an input file", accept_multiple_files=True, label_visibility="collapsed")

full_df = pd.DataFrame()

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        filename = parse_filename(uploaded_file.name)
        st.write("filename:", filename["mt"])
        st.write(df)

        #create a new df for manipulating
        new_df = df
        new_df.iloc[:,0] = new_df.iloc[:,0].str.removeprefix("result/$S_testbench/$RS_Testrig_output/$S_testbench.$X_")
        st.write(new_df)
        full_df = pd.concat([full_df, new_df], ignore_index= True)

        # create bar graph to display data
        #fig, ax = plt.subplots()

        #ax.plot(df.iloc[0].head(5), df.iloc[1].head(5))

        #st.pyplot(fig)

    st.write(full_df)

    #remove duplicates and sort list alphabetically
    axis_list = sorted(rem_duplicates(full_df.iloc[:,0]))

    #create a drop down to choose axis
    x_axis = st.sidebar.selectbox("Select X axis", axis_list)
    y_axis = st.sidebar.selectbox("Select Y axis", axis_list)

    #change index to axis names
    full_df.set_index('time', inplace=True)
    st.write(full_df)
    pog_df = full_df.loc[[x_axis, y_axis]]
    st.write(pog_df)





    


