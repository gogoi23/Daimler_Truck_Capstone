import pandas as pd
import numpy as np
import plotly.graph_objects as go
#import plotly.express as px
#from plotly.subplots import make_subplots

def plot(data, x_lim=None, y_lim=None, title=None,
         x_title=None, y_title=None, legends=None,
         quad1_title=None, quad2_title=None, 
         quad3_title=None, quad4_title=None):
    """
    Makes singular plot. Data is passed to function by alternating
    x- and y-parameters for each line. Data should be in DataFrame 
    with rows as seen below:
    |________________
    |line1_x | [...]
    |line1_y | [...]
    |line2_x | [...]
    |line2_y | [...]
    |...
    """
    
    #manipulate data into numpy array for each of plot construction
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)
    
    fig = go.Figure()
    num_lines = int(data.shape[0] / 2)
    #adding each line to the plot
    for i in range(int(num_lines)):
        fig.add_trace(
            go.Scatter(x=data[2*i], y=data[2*i + 1], mode='lines',
                       name=legends[i] if legends is not None else str(i))
        )
        
    #customize plot
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        xaxis_range=x_lim,
        yaxis_title=y_title,
        yaxis_range=y_lim
    )
    
    #add quadrant flags if provided, otherwise set default flags and hide them
    margins = 0.075
    fig.add_annotation(
        text=quad1_title if quad1_title is not None and quad1_title  else 'Click to enter Quadrant I flag',
        xref='paper', yref='paper',
        x=1-margins, y=1-margins, showarrow=False,
        opacity=1 if quad1_title is not None and quad1_title else 0
    )
    fig.add_annotation(
        text=quad2_title if quad2_title is not None and quad2_title else 'Click to enter Quadrant II flag',
        xref='paper', yref='paper',
        x=margins, y=1-margins, showarrow=False,
        opacity=1 if quad2_title is not None and quad2_title else 0
    )
    fig.add_annotation(
        text=quad3_title if quad3_title is not None and quad3_title else 'Click to enter Quadrant III flag',
        xref='paper', yref='paper',
        x=margins, y=margins, showarrow=False,
        opacity=1 if quad3_title is not None and quad3_title else 0
    )
    fig.add_annotation(
        text=quad4_title if quad4_title is not None and quad4_title else 'Click to enter Quadrant IV flag',
        xref='paper', yref='paper',
        x=1-margins, y=margins, showarrow=False,
        opacity=1 if quad4_title is not None and quad3_title else 0
    )
    
    return fig

def update(fig, x_lim=None, y_lim=None, 
           quad1_show=None, quad2_show=None, quad3_show=None, quad4_show=None,
           x_offsets=None, y_offsets=None, x_gridlines=True, y_gridlines=True,
           colors=None):
    """
    Update figure with with new customization if provided. 
    Capable of adjusting:
    -x and y-axes bounds
    -annotation/quadrant flag visibility
    -x- and y-directional offsets for individual traces/lines
    """
    
    #update flag visibility
    if quad1_show is not None:
        fig.update_layout(
            annotations=[dict(opacity=1 if quad1_show else 0),
                         dict(),
                         dict(),
                         dict()]
        )
    if quad2_show is not None:
        fig.update_layout(
            annotations=[dict(),
                         dict(opacity=1 if quad2_show else 0),
                         dict(),
                         dict()]
        )
    if quad3_show is not None:
        fig.update_layout(
            annotations=[dict(),
                         dict(),
                         dict(opacity=1 if quad3_show else 0),
                         dict()]
        )
    if quad4_show is not None:
        fig.update_layout(
            annotations=[dict(),
                         dict(),
                         dict(),
                         dict(opacity=1 if quad4_show else 0)]
        )
    
    #update x- and y-axes ranges
    if x_lim is not None:
        fig.update_xaxes(
            range=x_lim,
            showgrid=x_gridlines
        )
    if y_lim is not None:
        fig.update_yaxes(
            range=y_lim,
            showgrid=y_gridlines
        )
    
    #updates offsets of individual traces
    if x_offsets is not None:
        for trace, x_off in zip(fig.data, x_offsets):
            trace['x'] = np.add(trace['x'], x_off)
    if y_offsets is not None:
        for trace, y_off in zip(fig.data, y_offsets):
            trace['y'] = np.add(trace['y'], y_off)
            
    #updates colors of individual traces
    if colors is not None:
        for trace, color in zip(fig.data, colors):
            trace.line.color = color
        
    return fig

#for testing purposes
if __name__ == '__main__':
    data = pd.read_csv('.\Kinney_Exchange\Dummy Folder with FA1 results\output\sub_fa__hk_typ1__200415__P4_12p5k_ZZ1513_P__201003__01p00___sub_suspension_kinematics__210528__artic.csv')
    #fig = plot(data.iloc[:,1:], legends=data.iloc[:,0])
