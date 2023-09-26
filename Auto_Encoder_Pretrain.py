import math
from scipy.io import loadmat

import random
import torch
import torch.nn as nn

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
data_path = r'./dataset'
batch_size = 64
hidden_size = 200
dropout_ratio = 0.3
num_layers = 1
mylr = 0.0002
epochs = 50


# weight_decay = 1e-2


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
    filenames = os.walk(data_path)
    day_list = []

    for filepath, dirname, file_names in filenames:
        for file_name in file_names:
            if (file_name[-3:] == 'csv') & (file_name != 'time_traj.csv'):
                day_int = file_name.split('_')[-1].split('.')[0]
                day_list.append(day_int)

    day_list = list(set(day_list))
    train_date_list, test_date_list = train_test_split_func(day_list, ratio, shuffle=True)
    print(f'train_day_num:{len(train_date_list)},test_day_num:{len(test_date_list)}')

    return train_date_list, test_date_list


def norm_calcu_fuc():
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
    train_date_list, test_date_list = time_traj_list()
    filenames = os.walk(data_path)
    train_data_list = []
    test_data_list = []
    num_count_2 = 0
    for filepath, dirname, file_names in filenames:
        for file_name in file_names:
            if (file_name[-3:] == 'csv') & (file_name != 'time_traj.csv'):
                if file_name.split('_')[-1].split('.')[0] in train_date_list:
                    if '_' in file_name:
                        train_data_list.append('with_target/' + file_name)
                    else:
                        train_data_list.append('no_target/' + file_name)
                else:
                    if '_' in file_name:
                        test_data_list.append('with_target/' + file_name)
                    else:
                        test_data_list.append('no_target/' + file_name)
                num_count_2 += 1
                if num_count_2 % 10000 == 0:
                    print(f'************Reading data number: {num_count_2}************')
    train_data_pairs = []
    test_data_pairs = []

    for file_name in train_data_list:
        train_x = pd.read_csv(data_path + '/' + file_name).iloc[:, 1:]
        train_x = np.array(train_x)
        train_x = (train_x.transpose() - x_mean_value) / x_std_value
        train_x_y = [train_x.transpose(), train_x.transpose()]
        train_data_pairs.append(train_x_y)

    for file_name in test_data_list:
        test_x = pd.read_csv(data_path + '/' + file_name).iloc[:, 1:]
        test_x = np.array(test_x)
        test_x = (test_x.transpose() - x_mean_value) / x_std_value
        test_x_y = [test_x.transpose(), test_x.transpose()]
        test_data_pairs.append(test_x_y)
    print(f'train_set_num:{len(train_data_pairs)},test_set_num:{len(test_data_pairs)}')

    return train_data_pairs, test_data_pairs


start_time = time.time()
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

        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size, output_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, input, hidden, c):
        input = input.view(-1, 121, 25)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # rr = rr[:, -1, :]
        rr = self.dropout(rr)
        rr = self.linear1(rr)
        rr = rr.view(-1, 25, 121)
        # hn = hn[-1, :, :]
        # hn = self.linear2(hn)

        return rr, hn, c

    def inithiddenAndC(self):
        c = hidden = torch.zeros(num_layers, self.batch_size, self.hidden_size, device=device)
        return hidden, c


def Train():
    Encoder = EncoderLSTM(25, hidden_size, 25, num_layers, batch_size).to(device)
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

        Encoder.train()
        for train_item, (train_x, train_y) in enumerate(tqdm(train_dataloader), start=1):
            train_hidden, train_c = Encoder.inithiddenAndC()
            train_output, train_hidden, train_c = Encoder(train_x, train_hidden, train_c)
            train_output = train_output.to(device)
            # train_hidden = train_hidden.to(device)

            Encoder.zero_grad()
            train_loss = mse_loss(train_output, train_y)
            # train_loss = mse_loss(train_hidden.squeeze(), train_y)
            myadam_encode.zero_grad()
            train_loss.backward()
            myadam_encode.step()

            train_iter_num = train_iter_num + 1
            train_total_loss = train_total_loss + train_loss.item()

        Encoder.eval()
        with torch.no_grad():
            for test_item, (test_x, test_y) in enumerate(test_dataloader, start=1):
                test_hidden, test_c = Encoder.inithiddenAndC()
                test_predict, test_hidden, test_c = Encoder(test_x, test_hidden, test_c)
                test_loss = mse_loss(test_predict, test_y)
                # test_loss = mse_loss(test_hidden.squeeze(), test_y)

                test_iter_num = test_iter_num + 1
                test_total_loss = test_total_loss + test_loss.item()

        train_mse_loss = train_total_loss / train_iter_num
        test_mse_loss = test_total_loss / test_iter_num

        train_rmse_loss = math.sqrt(train_mse_loss)
        test_rmse_loss = math.sqrt(test_mse_loss)
        train_avg_loss_list.append(train_rmse_loss)
        test_avg_loss_list.append(test_rmse_loss)
        time.sleep(0.2)
        print('Epoch:', epoch_idx, "Train RMSELoss:", train_rmse_loss, "|", "Test RMSELoss:", test_rmse_loss)

    torch.save(Encoder.state_dict(), './model/anto_encoder_pretrain_%s.pth' % time.time())
    pd.DataFrame(train_avg_loss_list).to_csv('train_avg_loss_list.csv')
    pd.DataFrame(test_avg_loss_list).to_csv('test_avg_loss_list.csv')

    plt.figure()
    plt.plot(train_avg_loss_list, label='train_loss', color='b')
    plt.plot(test_avg_loss_list, label='test_loss', color='g')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./LSTM_RMSE_loss.png')
    plt.show()
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')


if __name__ == '__main__':
    Train()
