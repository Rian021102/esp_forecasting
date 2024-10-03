import pandas as pd
import stumpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt

def load_data(path, well_name, feat_name):
    df=pd.read_csv(path)
    #choose the well and feature
    df_feat = df[df['Well'] == well_name][['Well', 'Date', feat_name]]
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    return df_feat

def main():
    path='/Users/rianrachmanto/miniforge3/project/esp_new_02.csv'
    well_name = 'YWB-15'
    feat_name = 'Ampere'
    df_feat=load_data(path, well_name, feat_name)
    #find mofit using Stump
    m = 7
    mp = stumpy.stump(df_feat[feat_name], m)
    motif_idx = np.argsort(mp[:, 0])[0]
    print(f"The motif is located at index {motif_idx}")
    nearest_neighbor_idx = mp[motif_idx, 1]
    print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")
    ig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
    axs[0].plot(df_feat[feat_name].values)
    axs[0].set_ylabel(feat_name, fontsize='20')
    rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Date', fontsize ='20')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plt.show()
   



if __name__ == '__main__':
    main()
    
    
