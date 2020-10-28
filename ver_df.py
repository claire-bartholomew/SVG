import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt 

def main():
    df = pd.DataFrame([[0,0,0,0,0,0]], columns=['month','threshold','neighbourhood', 'flt', 'fss_nn', 'fss_p'])

    lists = [[1,1,25,15,0.87,0.87],
         [1,1,25,30,0.68,0.72],
         [1,1,25,45,0.56,0.62],
         [1,1,25,60,0.47,0.54],
         [1,2,25,15,0.80,0.81],
         [1,2,25,30,0.56,0.62],
         [1,2,25,45,0.42,0.49],
         [1,2,25,60,0.32,0.40],
         [1,4,25,15,0.68,0.71],
         [1,4,25,30,0.38,0.48],
         [1,4,25,45,0.24,0.35],
         [1,4,25,60,0.16,0.26],
         [1,8,25,15,0.43,0.47],
         [1,8,25,30,0.14,0.23],
         [1,8,25,45,0.05,0.14],
         [1,8,25,60,0.04,0.11],
         [2,1,25,15,0.86,0.86],
         [2,1,25,30,0.69,0.73],
         [2,1,25,45,0.56,0.64],
         [2,1,25,60,0.47,0.58],
         [2,2,25,15,0.79,0.79],
         [2,2,25,30,0.57,0.61],
         [2,2,25,45,0.43,0.50],
         [2,2,25,60,0.34,0.43],
         [2,4,25,15,0.64,0.65],
         [2,4,25,30,0.37,0.42],
         [2,4,25,45,0.24,0.30],
         [2,4,25,60,0.18,0.24],
         [2,8,25,15,0.40,0.42],
         [2,8,25,30,0.13,0.18],
         [2,8,25,45,0.07,0.10],
         [2,8,25,60,0.04,0.06],
         [3,1,25,15,0.86,0.87],
         [3,1,25,30,0.71,0.76],
         [3,1,25,45,0.60,0.70],
         [3,1,25,60,0.52,0.65],
         [3,2,25,15,0.79,0.81],
         [3,2,25,30,0.60,0.67],
         [3,2,25,45,0.47,0.60],
         [3,2,25,60,0.39,0.54],
         [3,4,25,15,0.67,0.71],
         [3,4,25,30,0.43,0.55],
         [3,4,25,45,0.31,0.47],
         [3,4,25,60,0.24,0.41],
         [3,8,25,15,0.45,0.53],
         [3,8,25,30,0.19,0.35],
         [3,8,25,45,0.11,0.29],
         [3,8,25,60,0.08,0.25],
         [4,1,25,15,0.89,0.89],
         [4,1,25,30,0.71,0.78],
         [4,1,25,45,0.59,0.70],
         [4,1,25,60,0.51,0.64],
         [4,2,25,15,0.84,0.84],
         [4,2,25,30,0.60,0.69],
         [4,2,25,45,0.44,0.59],
         [4,2,25,60,0.36,0.53],
         [4,4,25,15,0.72,0.73],
         [4,4,25,30,0.42,0.54],
         [4,4,25,45,0.26,0.44],
         [4,4,25,60,0.20,0.38],
         [4,8,25,15,0.46,0.47],
         [4,8,25,30,0.15,0.21],
         [4,8,25,45,0.08,0.16],
         [4,8,25,60,0.06,0.13],
         [5,1,25,15,0.86,0.88],
         [5,1,25,30,0.65,0.75],
         [5,1,25,45,0.53,0.65],
         [5,1,25,60,0.45,0.58],
         [5,2,25,15,0.78,0.81],
         [5,2,25,30,0.53,0.62],
         [5,2,25,45,0.40,0.49],
         [5,2,25,60,0.32,0.42],
         [5,4,25,15,0.66,0.69],
         [5,4,25,30,0.36,0.44],
         [5,4,25,45,0.25,0.31],
         [5,4,25,60,0.18,0.27],
         [5,8,25,15,0.50,0.55],
         [5,8,25,30,0.17,0.26],
         [5,8,25,45,0.10,0.16],
         [5,8,25,60,0.06,0.16],
         [6,1,25,15,0.92,0.92],
         [6,1,25,30,0.79,0.84],
         [6,1,25,45,0.68,0.77],
         [6,1,25,60,0.60,0.72],
         [6,2,25,15,0.87,0.87],
         [6,2,25,30,0.67,0.74],
         [6,2,25,45,0.52,0.65],
         [6,2,25,60,0.42,0.58],
         [6,4,25,15,0.79,0.78],
         [6,4,25,30,0.49,0.59],
         [6,4,25,45,0.29,0.47],
         [6,4,25,60,0.20,0.38],
         [6,8,25,15,0.67,0.66],
         [6,8,25,30,0.27,0.42],
         [6,8,25,45,0.09,0.28],
         [6,8,25,60,0.04,0.19],
         [7,1,25,15,0.87,0.88],
         [7,1,25,30,0.71,0.76],
         [7,1,25,45,0.58,0.67],
         [7,1,25,60,0.49,0.60],
         [7,2,25,15,0.80,0.83],
         [7,2,25,30,0.58,0.67],
         [7,2,25,45,0.44,0.57],
         [7,2,25,60,0.35,0.50],
         [7,4,25,15,0.65,0.73],
         [7,4,25,30,0.39,0.55],
         [7,4,25,45,0.26,0.45],
         [7,4,25,60,0.18,0.39],
         [7,8,25,15,0.45,0.54],
         [7,8,25,30,0.16,0.32],
         [7,8,25,45,0.07,0.24],
         [7,8,25,60,0.04,0.21],
         [8,1,25,15,0.81,0.84],
         [8,1,25,30,0.61,0.69],
         [8,1,25,45,0.49,0.60],
         [8,1,25,60,0.40,0.54],
         [8,2,25,15,0.76,0.79],
         [8,2,25,30,0.53,0.61],
         [8,2,25,45,0.39,0.50],
         [8,2,25,60,0.31,0.43],
         [8,4,25,15,0.66,0.73],
         [8,4,25,30,0.37,0.50],
         [8,4,25,45,0.24,0.38],
         [8,4,25,60,0.17,0.29],
         [8,8,25,15,0.46,0.60],
         [8,8,25,30,0.14,0.35],
         [8,8,25,45,0.07,0.23],
         [8,8,25,60,0.05,0.15],
         [9,1,25,15,0.83,0.87],
         [9,1,25,30,0.64,0.75],
         [9,1,25,45,0.53,0.66],
         [9,1,25,60,0.45,0.59],
         [9,2,25,15,0.76,0.83],
         [9,2,25,30,0.54,0.67],
         [9,2,25,45,0.41,0.56],
         [9,2,25,60,0.32,0.48],
         [9,4,25,15,0.63,0.76],
         [9,4,25,30,0.36,0.55],
         [9,4,25,45,0.23,0.42],
         [9,4,25,60,0.17,0.34],
         [9,8,25,15,0.41,0.64],
         [9,8,25,30,0.13,0.40],
         [9,8,25,45,0.06,0.27],
         [9,8,25,60,0.03,0.21],
         [10,1,25,15,0.86,0.87],
         [10,1,25,30,0.70,0.77],
         [10,1,25,45,0.60,0.70],
         [10,1,25,60,0.53,0.64],
         [10,2,25,15,0.79,0.82],
         [10,2,25,30,0.59,0.68],
         [10,2,25,45,0.48,0.58],
         [10,2,25,60,0.41,0.52],
         [10,4,25,15,0.65,0.71],
         [10,4,25,30,0.40,0.51],
         [10,4,25,45,0.29,0.40],
         [10,4,25,60,0.23,0.33],
         [10,8,25,15,0.45,0.52],
         [10,8,25,30,0.18,0.28],
         [10,8,25,45,0.11,0.17],
         [10,8,25,60,0.07,0.12],
         [11,1,25,15,0.91,0.91],
         [11,1,25,30,0.74,0.81],
         [11,1,25,45,0.60,0.72],
         [11,1,25,60,0.50,0.66],
         [11,2,25,15,0.86,0.87],
         [11,2,25,30,0.64,0.73],
         [11,2,25,45,0.47,0.64],
         [11,2,25,60,0.37,0.56],
         [11,4,25,15,0.75,0.78],
         [11,4,25,30,0.44,0.59],
         [11,4,25,45,0.27,0.47],
         [11,4,25,60,0.19,0.40],
         [11,8,25,15,0.55,0.64],
         [11,8,25,30,0.20,0.41],
         [11,8,25,45,0.08,0.28],
         [11,8,25,60,0.04,0.20],
         [12,1,25,15,0.84,0.84],
         [12,1,25,30,0.67,0.69],
         [12,1,25,45,0.55,0.60],
         [12,1,25,60,0.47,0.53],
         [12,2,25,15,0.77,0.77],
         [12,2,25,30,0.56,0.58],
         [12,2,25,45,0.42,0.48],
         [12,2,25,60,0.34,0.41],
         [12,4,25,15,0.64,0.65],
         [12,4,25,30,0.38,0.43],
         [12,4,25,45,0.26,0.33],
         [12,4,25,60,0.20,0.27],
         [12,8,25,15,0.44,0.45],
         [12,8,25,30,0.18,0.24],
         [12,8,25,45,0.09,0.15],
         [12,8,25,60,0.07,0.11]]

    df3 = pd.DataFrame(lists, columns=['month','threshold','neighbourhood', 'flt', 'fss_nn', 'fss_p'])
    thrshld = 1
    plot_fss(df3, thrshld)

def season_fss(df, thrshld, months, time):
    df4 = df.loc[df['threshold'] == thrshld]
    df5 = df4.loc[df4['month'].isin(months)]
    df6 = df5.loc[df5['flt'] == time]
    fss_n = np.mean(df6['fss_nn'])
    fss_p = np.mean(df6['fss_p'])
    return(fss_n, fss_p)

def plot_fss(df, thrshld):
    timestep = [15, 30, 45, 60]
    colors = ['blue', 'green', 'red', 'yellow']
    for nm, month in enumerate(['DJF','MAM','JJA','SON']):
        if month == 'DJF':
            months = [12, 1, 2]
        elif month == 'MAM':
            months = [3, 4, 5]
        elif month == 'JJA':
            months = [6, 7, 8]
        elif month == 'SON':
            months = [9, 10, 11]
        fss = []
        for time in timestep:
            fss_n, fss_p = season_fss(df, thrshld, months, time)
            fss.append(fss_n)
        plt.plot(timestep, fss, color=colors[nm], label=month)

    plt.xlabel('Time (minutes)')
    plt.ylabel('FSS')
    plt.legend(fontsize=10, ncol=4)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
