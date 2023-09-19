import math
# from scipy.io import loadmat

# from sklearn.preprocessing import StandardScaler
import random
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
ratio = 0.15
data_path = r'./dataset/with_target'
batch_size = 64
hidden_size = 100
dropout_ratio = 0.4
num_layers = 1
mylr = 0.003
epochs = 25


def train_test_split_func(full_list, ratio, shuffle=True):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    test_data_list = full_list[:offset]
    train_data_list = full_list[offset:]
    return train_data_list, test_data_list


def time_traj_list():
    time_traj_df = pd.read_csv('./dataset/time_traj.csv')
    date_str_list = []
    date_dic = {}
    for idx in range(len(time_traj_df)):
        date_str = str(time_traj_df.iloc[idx][1]) + str(time_traj_df.iloc[idx][2]) + str(time_traj_df.iloc[idx][3])
        time_str = str(time_traj_df.iloc[idx][-1])
        date_str_list.append(date_str)
        date_dic[time_str] = date_str
        if idx % 20000 == 0:
            print(f'************Reading data number: {idx}************')
    date_str_list = list(set(date_str_list))
    train_date_list, test_date_list = train_test_split_func(date_str_list, ratio, shuffle=True)
    return train_date_list, test_date_list, date_dic


def norm_calcu_fuc():
    # filename = 'Converted_data_for_AE'
    # raw_data = loadmat(filename)
    # feature = raw_data['Input_time_series']
    # np_tmp = feature[0][0]
    # for i in range(1, feature.size):
    #     np_tmp = np.concatenate([np_tmp, feature[i][0]], axis=1)
    #     if i % 2000 == 0:
    #         print(f'************num:{i}************')
    # x_mean_value = np.mean(np_tmp, axis=1)
    # x_std_value = np.std(np_tmp, axis=1)
    # x_std_value[x_std_value == 0] = 1
    x_mean_value = [9.06558217e+02, 1.90736501e-01, 3.60037723e+00, 7.23522534e-02,
                    3.60786419e-01, 1.42427916e-01, 1.62456484e+02, 1.01914306e+05,
                    4.94778053e-01, 6.09212095e+00, 1.69291848e+01, 2.82171400e+02,
                    1.13739105e-02, 1.38764537e-02, 4.23811116e-02, 7.34597943e-01,
                    2.77331227e+02, 3.73565421e-07, 3.78497820e-02, 4.79864031e-03,
                    3.12594811e-03, 3.53839895e-05, 8.02430340e-04, 1.77910194e+00,
                    3.20613748e-01]

    x_std_value = [1.37970459e+02, 3.85012200e-01, 6.51165494e-01, 3.94930877e-02,
                   3.33266285e-01, 2.57680915e-01, 2.44643838e+02, 9.75667198e+02,
                   1.78784409e-01, 3.35759877e+00, 5.80827754e+00, 1.50634956e+01,
                   2.88157752e-03, 9.12916032e-02, 4.90390310e-02, 2.04382989e-01,
                   1.69076187e+01, 1.20099042e-06, 1.42534801e-03, 1.60366323e-03,
                   1.38708423e-03, 5.02354607e-04, 1.36736020e-03, 1.26844302e+00,
                   3.21069745e-01]

    return x_mean_value, x_std_value


def my_getdata():
    x_mean_value, x_std_value = norm_calcu_fuc()
    train_date_list, test_date_list, date_dic = time_traj_list()
    filenames = os.listdir(data_path)
    train_data_list = []
    test_data_list = []
    for file_name in filenames:
        if date_dic[file_name.split('_')[1].strip('.csv')] in train_date_list:
            train_data_list.append(file_name)
        else:
            test_data_list.append(file_name)
    train_data_pairs = []
    test_data_pairs = []

    for file_name in train_data_list:
        train_x = pd.read_csv(data_path + '/' + file_name).iloc[:, 1:]
        # train_y = float(file_name.strip('.csv')[0])
        train_y = float(file_name.split('_')[0])
        train_x = np.array(train_x)
        train_x = (train_x.transpose() - x_mean_value) / x_std_value
        train_x_y = [train_x.transpose(), train_y]
        train_data_pairs.append(train_x_y)

    for file_name in test_data_list:
        test_x = pd.read_csv(data_path + '/' + file_name).iloc[:, 1:]
        test_y = float(file_name.split('_')[0])
        # test_y = float(file_name.strip('.csv')[0])
        test_x = np.array(test_x)
        test_x = (test_x.transpose() - x_mean_value) / x_std_value
        test_x_y = [test_x.transpose(), test_y]
        test_data_pairs.append(test_x_y)
    return train_data_pairs, test_data_pairs


train_data_pairs, test_data_pairs = my_getdata()


class DataPairsDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        self.sample_len = len(data_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(max(index, 0), self.sample_len - 1)

        x = self.data_pairs[index][0]
        y = self.data_pairs[index][1]

        tensor_x = torch.tensor(x, dtype=torch.float, device=device)
        tensor_y = torch.tensor(y, dtype=torch.float, device=device)

        return tensor_x, tensor_y


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, input, hidden, c):
        input = input.view(batch_size, 121, 25)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        rr = rr[:, -1, :]
        rr = self.linear(rr)
        # rr = self.dropout(rr)
        return rr, hn, c

    def inithiddenAndC(self):
        c = hidden = torch.zeros(num_layers, self.batch_size, self.hidden_size, device=device)
        return hidden, c


def Train():
    start_time = time.time()
    Encoder = EncoderLSTM(25, hidden_size, 1, num_layers, batch_size).to(device)
    myadam_encode = torch.optim.Adam(Encoder.parameters(), lr=mylr)
    mse_loss = nn.MSELoss()

    train_avg_loss_list = []
    test_avg_loss_list = []

    for epoch_idx in range(1, epochs + 1):

        train_iter_num = 0
        test_iter_num = 0
        train_total_loss = 0
        test_total_loss = 0
        train_dataset = DataPairsDataset(train_data_pairs)
        test_dataset = DataPairsDataset(test_data_pairs)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        with torch.no_grad():
            for test_item, (test_x, test_y) in enumerate(test_dataloader, start=1):
                test_hidden, test_c = Encoder.inithiddenAndC()
                test_predict, test_hidden, test_c = Encoder(test_x, test_hidden, test_c)
                test_loss = mse_loss(test_predict.squeeze(), test_y)

                test_iter_num = test_iter_num + 1
                test_total_loss = test_total_loss + test_loss.item()

        for train_item, (train_x, train_y) in enumerate(tqdm(train_dataloader), start=1):
            train_hidden, train_c = Encoder.inithiddenAndC()
            train_output, train_hidden, train_c = Encoder(train_x, train_hidden, train_c)
            train_output = train_output.to(device)

            Encoder.zero_grad()
            train_loss = mse_loss(train_output.squeeze(), train_y)
            myadam_encode.zero_grad()
            train_loss.backward()
            myadam_encode.step()

            train_iter_num = train_iter_num + 1
            train_total_loss = train_total_loss + train_loss.item()

        train_mse_loss = train_total_loss / train_iter_num
        test_mse_loss = test_total_loss / test_iter_num

        train_rmse_loss = math.sqrt(train_mse_loss)
        test_rmse_loss = math.sqrt(test_mse_loss)
        train_avg_loss_list.append(train_rmse_loss)
        test_avg_loss_list.append(test_rmse_loss)
        print('Epoch:', epoch_idx, "Train RMSELoss:", train_rmse_loss, "|", "Test RMSELoss:", test_rmse_loss)

    torch.save(Encoder.state_dict(), './model/lstm_method_%s.pth' % time.time())
    pd.DataFrame(train_avg_loss_list).to_csv('train_avg_loss_list.csv')
    pd.DataFrame(test_avg_loss_list).to_csv('test_avg_loss_list.csv')
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')
    plt.figure()
    plt.plot(train_avg_loss_list, label='train_loss', color='b')
    plt.plot(test_avg_loss_list, label='test_loss', color='g')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./LSTM_RMSE_loss.png')
    plt.show()


if __name__ == '__main__':
    Train()
