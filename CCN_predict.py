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
data_path = r'./fake_data'
batch_size = 2
hidden_size = 16
num_layers = 1
mylr = 0.01
epochs = 100


def my_getdata():
    filenames = os.listdir(data_path)
    data_pairs = []
    for file_name in filenames:
        x = pd.read_csv('./fake_data/' + file_name).iloc[:, 1:]
        y = float(file_name.strip('.csv'))
        x_y = [np.array(x), y]
        data_pairs.append(x_y)

    return data_pairs


data_pairs = my_getdata()


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
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, c):
        input = input.view(batch_size, 121, 25)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        rr = rr[:, -1, :]
        rr = self.linear(rr)
        return rr, hn, c

    def inithiddenAndC(self):
        c = hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
        return hidden, c


def Train():
    start_time = time.time()
    Encoder = EncoderLSTM(25, hidden_size, 1, num_layers, batch_size).to(device)
    myadam_encode = torch.optim.Adam(Encoder.parameters(), lr=mylr)
    mse_loss = nn.MSELoss()

    total_iter_num = 0
    total_loss = 0.0
    total_loss_list = []

    for epoch_idx in range(1, 1 + epochs):
        mypairsdataset = DataPairsDataset(data_pairs)
        mydataloader = DataLoader(dataset=mypairsdataset, batch_size=batch_size, shuffle=True)
        for item, (x, y) in enumerate(tqdm(mydataloader), start=1):
            hidden, c = Encoder.inithiddenAndC()

            output, hidden, c = Encoder(x, hidden, c)
            output = output.to(device)
            Encoder.zero_grad()
            loss = mse_loss(output.squeeze(), y)
            myadam_encode.zero_grad()
            loss.backward()
            myadam_encode.step()

            total_iter_num = total_iter_num + 1
            total_loss = total_loss + loss.item()

            if (total_iter_num % 10 == 0):
                tmploss = total_loss / total_iter_num * batch_size
                total_loss_list.append(tmploss)

    torch.save(Encoder.state_dict(), './model/lstm_method_%s.pth' % time.time())
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')
    plt.figure()
    plt.plot(total_loss_list)
    plt.savefig('./RMSE_loss.png')
    plt.show()


if __name__ == '__main__':
    Train()
