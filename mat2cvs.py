from scipy.io import loadmat
import pandas as pd
import numpy as np

import time
import torch

filename = 'Converted_data_for_AE'
raw_data = loadmat(filename)

feature = raw_data['Input_time_series']
Target_CCN = raw_data['Target_CCN']
time_traj = raw_data['time_traj']


def mat_to_cvs():
    for idx in range(len(time_traj)):
        if str(Target_CCN[idx].item()) == 'nan':
            filename = str(time_traj[idx][-1]) + '.csv'
            pd.DataFrame(feature[idx][0]).to_csv('./dataset/no_target/' + filename)
        else:
            filename = str(Target_CCN[idx].item()) + '_' + str(time_traj[idx][-1]) + '.csv'
            pd.DataFrame(feature[idx][0]).to_csv('./dataset/with_target/' + filename)


def time_traj_form():
    time_traj_df = pd.DataFrame(time_traj)
    # print(time_traj_df)
    time_traj_df.to_csv('./dataset/time_traj.csv')


def norm_calcu_fuc():
    np_tmp = feature[0][0]
    # for i in range(1, 30000):
    #     np_tmp = np.concatenate([np_tmp, feature[i][0]], axis=1)
    #     if i % 2000 == 0:
    #         print(f'************num:{i}************')
    # x_mean_value = np.mean(np_tmp, axis=1)
    # x_std_value = np.std(np_tmp, axis=1)

    feature_df = pd.DataFrame(list([list(feature[i]) for i in range(feature.size)]))
    mean_list = []
    std_list = []
    for i in range(25):
        list_i = []
        for j in range(68016):
            list_i.extend(list(feature_df[i][j]))
        array_i = np.array(list_i)
        mean_list.append(np.mean(array_i))
        std_list.append(np.std(array_i))

    return mean_list, std_list


if __name__ == '__main__':
    # mat_to_cvs()
    # time_traj_form()
    x_mean_value, x_std_value = norm_calcu_fuc()
    print(x_mean_value)
    print(x_std_value)
