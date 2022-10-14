import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(data, x_lim=None, y_lim=None, title=None, xtitle=None, ytitle=None):
    if not type(data).__module__ == np.__name__:
        data = np.array(data.values)
        
    X = data[0]
    if x_lim == None:
        x_lim = [np.min(X), np.max(X)]
    Y = data[1:6]
    print(len(Y))
    num_graphs = len(Y)
    fig, axs = plt.subplots(num_graphs, 1, sharex=True, sharey=False)
    for i, y_ax in enumerate(Y):
        axs[i].set_xlim(x_lim)
        axs[i].set_ylim(y_lim if y_lim != None else [np.min(y_ax),np.max(y_ax)])
        axs[i].plot(X, y_ax)
    #plt.show()

    return fig
    
if __name__ == '__main__':
    data = pd.read_csv('.\Kinney_Exchange\Dummy Folder with FA1 results\output\sub_fa__hk_typ1__200415__P4_12p5k_ZZ1513_P__201003__01p00___sub_suspension_kinematics__210528__artic.csv')
    data = np.array(data.values)[:,1:]
    plot(data)
