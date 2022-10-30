import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#---------ARCHIVED FOR NOW-------------
#Makes subplots using to first row of data as the x-values for all of them
#and the successive rows of data as the y-values of each subplot
def plot_subplots(data, x_lim=None, y_lim=None,
                  title=None, xtitle=None, ytitle=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)
        
    X = data[0]
    Y = data[1:]
    fig = make_subplots(rows = Y.shape[0], cols = 1)
    
    #making seperate subplots for each line
    for i, y_ax in enumerate(Y):
        fig.add_trace(
            go.Scatter(x=X, y=y_ax, mode='lines', name=str(i)),
            row = i+1, col = 1
        )
        #customizing subplot
        #fig.update_xaxes(title_text=xtitle, range=x_lim, row=i+1, col=1)
        #fig.update_yaxes(title_text=ytitle, range=y_lim, row=i+1, col=1)
    return fig

#Makes makes singular plot using the first row of data as the x-values
#and the successive rows of data as different lines on same plot
def plot(data, x_lim=None, y_lim=None, title=None,
         x_title=None, y_title=None, legends=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)
    
    X = data[0]
    Y = data[1:]
    fig = go.Figure()
    #adding each line to the plot
    for i, y_ax in enumerate(Y):
        fig.add_trace(
            go.Scatter(x=X, y=y_ax, mode='lines',
                       name=legends[i] if legends is not None else None)
        )
        
    #customize plot
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        xaxis_range=x_lim,
        yaxis_title=y_title,
        yaxis_range=y_lim
        #,legend_title=""
    )
    
    #fig.show()
    return fig

#data must be pandas.DataFrame
#Same functionality as plot()
def plot_px(data, index_as_legends=True, x_lim=None, y_lim=None,
            title=None, x_title=None, y_title=None):
    #if the index column should not be the legend names, use the first column instead
    if not index_as_legends:
        data.set_index(data.columns[0], inplace=True)
    data = data.T
    
    #making the single plot with multiple lines
    fig = px.line(data, x=data.columns[0], y=data.columns[1:],
                  range_x=x_lim, range_y=y_lim, title=title)
    
    #customizing plot
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title
        #,legend_title=""
    )
    
    #fig.show()
    return fig

#for testing purposes
if __name__ == '__main__':
    data = pd.read_csv('.\Kinney_Exchange\Dummy Folder with FA1 results\output\sub_fa__hk_typ1__200415__P4_12p5k_ZZ1513_P__201003__01p00___sub_suspension_kinematics__210528__artic.csv')
    plot(data.iloc[:,1:], legends=data.iloc[:,0])
    plot_px(data, index_as_legends=True)
    
