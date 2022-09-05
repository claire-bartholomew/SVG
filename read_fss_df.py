import numpy as np
import pdb
from datetime import datetime, timedelta
import os
import iris
import matplotlib
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import nimrod_to_cubes as n2c
import pandas as pd

def main():

    filename = 'fss_df/daily_fss4_30_n9_th1.csv'
    df = pd.read_csv(filename)

    df = df.sort_values('fss_e_mean')
    print(df)
    print(df['fss_e_mean'])

    df_0p1 = df.loc[df['mean_data'] >= 0.1]

    print(df_0p1)

    pdb.set_trace()

    print(df_0p1.loc[df_0p1['fss_e_mean'] > df_0p1['fss_on']])


    ax2 = plt.gca()
    ax2 = df_0p1.plot.scatter(x='fss_on', y='fss_e_mean', c='mean_data', norm=matplotlib.colors.LogNorm(), cmap='jet') #Blues_r')
    #ax2 = df_0p1.loc[df_0p1['fss_e_mean'] > 0.1].plot.scatter(x='fss_on', y='fss_e_mean') #, c='mean_data') #color='b')
    #ax2 = df_0p1.loc[df_0p1['fss_e_mean'] < 0.1].plot.scatter(x='fss_on', y='fss_e_mean') #, c='mean_data') #color='r')
    ax2.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.show()

    pdb.set_trace()

    poor_cases = df_0p1.loc[df_0p1['fss_e_mean'] < 0.1]
    poor_dates = poor_cases['datetime']
    print(poor_dates)
    poor_cases.to_csv('poor_cases.csv', index=False)
    poor_dates.to_csv('poor_dates.csv', index=False)
    pdb.set_trace()

    df_0p1_lowfss = df_0p1.loc[df_0p1['fss_e_mean'] <= 0.2]

    df_0p1_lowfss['min_fss'] = df_0p1_lowfss[['fss_e0',
                        'fss_e1', 'fss_e2', 'fss_e3', 'fss_e4', 'fss_e5', 'fss_e6',
                        'fss_e7', 'fss_e8', 'fss_e9', 'fss_e10', 'fss_e11', 'fss_e12',
                        'fss_e13', 'fss_e14', 'fss_e15', 'fss_e16', 'fss_e17',
                        'fss_e18', 'fss_e19', 'fss_e20', 'fss_e21', 'fss_e22',
                        'fss_e23', 'fss_e24', 'fss_e25', 'fss_e26', 'fss_e27',
                        'fss_e28', 'fss_e29']].min(axis=1)
    df_0p1_lowfss['max_fss'] = df_0p1_lowfss[['fss_e0',
                        'fss_e1', 'fss_e2', 'fss_e3', 'fss_e4', 'fss_e5', 'fss_e6',
                        'fss_e7', 'fss_e8', 'fss_e9', 'fss_e10', 'fss_e11', 'fss_e12',
                        'fss_e13', 'fss_e14', 'fss_e15', 'fss_e16', 'fss_e17',
                        'fss_e18', 'fss_e19', 'fss_e20', 'fss_e21', 'fss_e22',
                        'fss_e23', 'fss_e24', 'fss_e25', 'fss_e26', 'fss_e27',
                        'fss_e28', 'fss_e29']].max(axis=1)

    df_0p1_lowfss['max-min'] = df_0p1_lowfss['max_fss'] - df_0p1_lowfss['min_fss']
    # ax3 = df_0p1_lowfss.plot.scatter(x='fss_e_mean', y='max_fss')
    # plt.show()
    #
    # ax4 = df_0p1_lowfss.plot.scatter(x='fss_e_mean', y='min_fss')
    # plt.show()

    #ax = plt.gca()
    #df_0p1_lowfss.plot(kind='scatter',x='fss_e_mean',y='max_fss',ax=ax)
    #df_0p1_lowfss.plot(kind='scatter',x='fss_e_mean',y='min_fss', color='red', ax=ax)
    #plt.show()

    df3 = df_0p1_lowfss.sort_values('max-min')

    pdb.set_trace()


if __name__ == "__main__":
    main()
