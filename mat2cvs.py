from scipy.io import loadmat
import pandas as pd


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

if __name__ == '__main__':
    # mat_to_cvs()
    time_traj_form()
