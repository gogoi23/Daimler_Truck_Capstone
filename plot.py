import pandas as pd
import numpy as np
import plotly.graph_objects as go
#import plotly.express as px
#from plotly.subplots import make_subplots

"""
Makes makes singular plot using the first row of data as the x-values
and the successive rows of data as different lines on same plot
Data should be in DataFrame with rows as seen below:
|________________
|line1_x | [...]
|line1_y | [...]
|line2_x | [...]
|line2_y | [...]
|...
"""
def plot(data, x_lim=None, y_lim=None, title=None,
         x_title=None, y_title=None, legends=None,
         quad1_title=None, quad2_title=None, 
         quad3_title=None, quad4_title=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)
    
    fig = go.Figure()
    num_lines = int(data.shape[0] / 2)
    #adding each line to the plot
    for i in range(num_lines):
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
    
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=2
    # )
    #fig['layout'].update(scene=dict(aspectmode="data"))
    
    #add quadrant flags if provided, otherwise set default flags and hide them
    margins = 0.075
    fig.add_annotation(
        text=quad1_title if quad1_title else 'Click to enter Quadrant I flag',
        xref='paper', yref='paper',
        x=1-margins, y=1-margins, showarrow=False,
        opacity=1 if quad1_title is not None else 0
    )
    fig.add_annotation(
        text=quad2_title if quad2_title else 'Click to enter Quadrant II flag',
        xref='paper', yref='paper',
        x=margins, y=1-margins, showarrow=False,
        opacity=1 if quad2_title is not None else 0
    )
    fig.add_annotation(
        text=quad3_title if quad3_title else 'Click to enter Quadrant III flag',
        xref='paper', yref='paper',
        x=margins, y=margins, showarrow=False,
        opacity=1 if quad3_title is not None else 0
    )
    fig.add_annotation(
        text=quad4_title if quad4_title else 'Click to enter Quadrant IV flag',
        xref='paper', yref='paper',
        x=1-margins, y=margins, showarrow=False,
        opacity=1 if quad4_title is not None else 0
    )
    
    #fig.show()
    return fig


"""
Update figure with with new customization if provided. 
Capable of adjusting:
-x and y-axes bounds
-annotation/quadrant flag visibility
-x and y-directional offsets for individual traces/lines
"""
def update(fig, x_lim=None, y_lim=None, 
           quad1_show=False, quad2_show=False, quad3_show=False, quad4_show=False,
           x_offsets=None, y_offsets=None, x_gridlines=True, y_gridlines=True):

    
    fig.update_layout(
        #title=dict(x=0.5, y=0.9),
        annotations=[
            dict(opacity=1 if quad1_show else 0),
            dict(opacity=1 if quad2_show else 0),
            dict(opacity=1 if quad3_show else 0),
            dict(opacity=1 if quad4_show else 0)
        ]
        #,legend_title=""
    )
    
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
    for trace, x_off, y_off in zip(fig.data, x_offsets, y_offsets):
        trace['x'] = np.add(trace['x'], x_off)
        trace['y'] = np.add(trace['y'], y_off)
    return fig

#updates figure with many different options
def update(fig, title=None, x_lim=None, y_lim=None, x_title=None, y_title=None,
           quad1_title=None, quad2_title=None, quad3_title=None, quad4_title=None,
           x_offset=0, y_offset=0, x_gridlines=True, y_gridlines=True):
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title,
                   range=x_lim,
                   showgrid=x_gridlines),
        yaxis=dict(title=y_title,
                   range=y_lim,
                   showgrid=y_gridlines)
        #,legend_title=""
    )
    
    margin = 0.075
    fig.add_annotation(
        text=quad1_title,
        xref='paper', yref='paper',
        x=1-margin, y=1-margin,showarrow=False
    )
    fig.add_annotation(
        text=quad2_title,
        xref='paper', yref='paper',
        x=margin, y=1-margin,showarrow=False
    )
    fig.add_annotation(
        text=quad3_title,
        xref='paper', yref='paper',
        x=margin, y=margin,showarrow=False
    )
    fig.add_annotation(
        text=quad4_title,
        xref='paper', yref='paper',
        x=1-margin, y=margin,showarrow=False
    )
    
    for trace in fig.data:
        
        trace['x'] = np.add(trace['x'], x_offset)
        trace['y'] = np.add(trace['y'], y_offset)
        
    return fig

#for testing purposes
if __name__ == '__main__':
    data = pd.read_csv('.\Kinney_Exchange\Dummy Folder with FA1 results\output\sub_fa__hk_typ1__200415__P4_12p5k_ZZ1513_P__201003__01p00___sub_suspension_kinematics__210528__artic.csv')

    #fig = plot(data.iloc[:,1:], legends=data.iloc[:,0])

    #plot_px(data, index_as_legends=True)
