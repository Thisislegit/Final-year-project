import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob
import pickle


def read_label(house_list):
    label = {}
    for i in house_list:
        hi = 'C:/Users/admin/Desktop/REDD_dataset/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def read_merge_data(house, labels):
    path = 'C:/Users/admin/Desktop/REDD_dataset/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df

def create_house_dataframe(house_list):
    labels = read_label(house_list)
    df = {}
    for i in house_list:
        df[i] = read_merge_data(i, labels)
        print("House {} finish:".format(i))
        print(df[i].head())

    return df

def date(house_list, df):
    dates = {}
    for i in house_list:
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i, len(dates[i]), dates[i][0], dates[i][-1]))
        print(dates[i], '\n')

    return dates

def save_main_meter(df, house_list = [1,2,3,4,5,6]):
    for i in house_list:
        #save X data from house 1 to 6
        X = df[i][['mains_1','mains_2']].values
        pickle_out = open("house_{}_main.pickle".format(i), "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()
        print("House {} main meter finish saving.".format(i))


def save_app_data(df, house, applicance):

    y = df[house][applicance].values
    pickle_out = open("house_{}_{}.pickle".format(house, applicance.split('_')[0]), "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def window_sliding(X, window_size):
    # first padding on dataset
    padding = np.zeros((window_size - 1, X.shape[1]))
    X1 = np.append(padding, X, axis=0)
    all_x_train = np.empty((X.shape[0], window_size, X.shape[1]))
    for idx in range(window_size, X1.shape[0] + 1):
        all_x_train[idx - window_size, :, :] = np.reshape(X1[idx - window_size:idx, :],
                                                            (1, window_size, X.shape[1]))
    return all_x_train


def plot_energy(df, house_list = [1,2]):
    # Plot total energy sonsumption of each appliance from two houses
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    plt.suptitle('Total enery consumption of each appliance', fontsize=30)
    cons1 = df[1][df[1].columns.values[2:]].sum().sort_values(ascending=False)
    app1 = cons1.index
    y_pos1 = np.arange(len(app1))
    axes[0].bar(y_pos1, cons1.values, alpha=0.6)
    plt.sca(axes[0])
    plt.xticks(y_pos1, app1, rotation=45)
    plt.title('House 1')

    cons2 = df[2][df[2].columns.values[2:]].sum().sort_values(ascending=False)
    app2 = cons2.index
    y_pos2 = np.arange(len(app2))
    axes[1].bar(y_pos2, cons2.values, alpha=0.6)
    plt.sca(axes[1])
    plt.xticks(y_pos2, app2, rotation=45)
    plt.title('House 2')

if __name__ == "__main__":
    house_list = [1,2,3,4,5,6]
    df = create_house_dataframe(house_list)
    dates = date(house_list, df)

    #Save data
    training_house = [1,2,3,4,5,6]
    test_house = [5]
    app = "refrigerator"
    save_main_meter(df, training_house)
    save_app_data(df, 1, "refrigerator_5")
    save_app_data(df, 2, "refrigerator_9")
    save_app_data(df, 5, "refrigerator_18")

    #Sliding Window
    pickle_in = open("house_1_main.pickle", "rb")
    X1 = pickle.load(pickle_in)
    pickle_in = open("house_2_main.pickle", "rb")
    X2 = pickle.load(pickle_in)
    pickle_in = open("house_5_main.pickle", "rb")
    X5 = pickle.load(pickle_in)
    pickle_in = open("house_1_refrigerator.pickle", "rb")
    y1 = pickle.load(pickle_in)
    pickle_in = open("house_2_refrigerator.pickle", "rb")
    y2 = pickle.load(pickle_in)
    pickle_in = open("house_5_refrigerator.pickle", "rb")
    y5 = pickle.load(pickle_in)

    X1_window = window_sliding(X1, 50)
    X2_window = window_sliding(X2, 50)
    X5_window = window_sliding(X5, 50)
    y1_window = window_sliding(y1.reshape(-1, 1), 50)
    y2_window = window_sliding(y2.reshape(-1, 1), 50)
    y5_window = window_sliding(y5.reshape(-1, 1), 50)

    X_train = np.append(X1_window, X2_window, axis=0)
    y_train = np.append(y1_window, y2_window, axis=0)
    X_test = X5_window
    y_test = y5_window

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    #Save data
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()
    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()
    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()
    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()