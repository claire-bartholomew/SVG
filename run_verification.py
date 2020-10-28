import verification as ver
import pandas as pd
import pdb

def main():
    # Select model number for running verification
    model_n = '624800' #' #625308' #624800' #'131219'
    # Choose variables
    #thrshld = 1 # rain rate threshold (mm/hr)
    neighbourhood = 25 #25   # neighbourhood size (e.g. 9 = 3x3)

    #combine_df(neighbourhood)

    #df = pd.DataFrame([[0,0,0,0,0,0]], columns=['month','threshold','neighbourhood', 'flt', 'fss_nn', 'fss_p'])
    lists = []
    # Run verification scripts to get FSS results
    for month in range(12, 13):
        for thrshld in [1,2,4,8]:
            for timesteps in [[15], [30], [45], [60]]:
                #print(thrshld)
                flt = timesteps[0]
                fss_nn, fss_p = ver.main(model_n, thrshld, neighbourhood, month, timesteps)
                list = [month, thrshld, neighbourhood, flt, fss_nn, fss_p]
                print(list)
                lists.append(list)
                #df2 = pd.DataFrame([list], columns=['month','threshold','neighbourhood', 'flt', 'fss_nn', 'fss_p'])
                #df.append(df2)

    df3 = pd.DataFrame(lists, columns=['month','threshold','neighbourhood', 'flt', 'fss_nn', 'fss_p'])
    df3.to_csv('df.out', index=False)

    #combine_df(neighbourhood)

def combine_df(neighbourhood):
    frames = []
    for month in range(1, 13):
        for thrshld in [1,2,4,8]:
            for timesteps in [[15], [30], [45], [60]]:
                df = pd.read_csv('fss_df_mon{}_t{}_{}mmhr_n{}.csv'.format(month, timesteps[0], thrshld, neighbourhood))
                frames.append(df)
    pdb.set_trace()

if __name__ == "__main__":
    main()
