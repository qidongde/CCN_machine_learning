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
mylr = 0.005
epochs = 20


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
    # feature_df = pd.DataFrame(list([list(feature[i]) for i in range(feature.size)]))
    # mean_list = []
    # std_list = []
    # for i in range(25):
    #     list_i = []
    #     for j in range(68016):
    #         list_i.extend(list(feature_df[i][j]))
    #     array_i = np.array(list_i)
    #     mean_list.append(np.mean(array_i))
    #     std_list.append(np.std(array_i))
    x_std_value = [146.14018088951715,
                   0.3891995865484179,
                   0.6574776979871536,
                   0.04004388611796154,
                   0.33145931756403924,
                   0.26364274030064555,
                   243.56554081478887,
                   993.3455215393174,
                   0.18208874564370145,
                   3.403268177459035,
                   5.944552578089976,
                   15.068052659958578,
                   0.0029259483770218155,
                   0.0988247533942239,
                   0.0502647129080475,
                   0.22107641140288598,
                   17.198015903422046,
                   1.063766861764658e-06,
                   0.001660794754560507,
                   0.0016132797275642075,
                   0.0013940533233640668,
                   0.0005607768697634261,
                   0.0013793804321126691,
                   1.301052106034106,
                   0.3207553288210386]

    x_mean_value = [883.4398482564153,
                    0.1965733390879346,
                    3.606625818074601,
                    0.07187857203492816,
                    0.3616113593868467,
                    0.14948980961002956,
                    161.50385094725152,
                    101881.39958631997,
                    0.493645780983079,
                    6.154375899871608,
                    16.83131248021451,
                    282.0303233806911,
                    0.011804987329518625,
                    0.016493584649592174,
                    0.045869369116387296,
                    0.7213918549269006,
                    276.05802689478605,
                    4.136759972442086e-07,
                    0.03821588070891682,
                    0.004802251454349955,
                    0.0031218371512679146,
                    4.7307741188862164e-05,
                    0.0008178438606103292,
                    1.6903382500290134,
                    0.3281926842969482]

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
