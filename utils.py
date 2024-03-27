import pandas as pd
import random, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import fftpack
import torch.utils.data as utils
import os, glob
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import numpy as np

class TrafficForecast(Dataset):
    def __init__(self,pkl_path, window, horizon,type):
        self.pkl_path = pkl_path #pkl path for train test predict
        self.window = window
        self.horizon = horizon
        self.df = None
        self.seg_ids = None
        self.inputs = []
        self.targets = []
        self.type = type
        self.train_df=None
        self.test_df=None


        df = pd.read_pickle(self.pkl_path)
        df['time'] = pd.to_datetime(df['time'])
        df['unix_timestamp'] = df['time'].astype(int) / 10**9
        #split into 80% train 20% test for each unique segid
        self.df = df.sort_values(by='time', ascending=True)
        self.seg_ids = self.df['segmentID'].unique()
        self.train_df, self.test_df = self.data_split()

        self.setup_forecast()

    def data_split(self):
        tr_df=[]
        test_df=[]
        for segid in self.seg_ids:         
            df_seg_id = self.df[self.df['segmentID'] == segid]    
            train_len = int(0.8 * len(df_seg_id))
            train_dfs = df_seg_id.iloc[:train_len]
            test_dfs = df_seg_id.iloc[train_len:]
            tr_df.append(train_dfs)
            test_df.append(test_dfs)
        return pd.concat(tr_df),pd.concat(test_df)

    def setup_forecast(self):
        if self.type == 'train':
            self.df = self.train_df
            print(len(self.df))
        elif self.type == 'test':
            self.df = self.test_df
            print(len(self.df))
        for segid in self.seg_ids:
            df_seg_id = self.df[self.df['segmentID'] == segid]
            df_seg_id = df_seg_id.fillna(method="ffill")
            df_seg_id = df_seg_id.sort_values(by='time', ascending=True)
            df_seg_id['hour'] = df_seg_id['time'].dt.hour
            df_seg_id['min'] = df_seg_id['time'].dt.minute
            df_seg_id['dow'] = df_seg_id['time'].dt.weekday
            TI_series = df_seg_id['TrafficIndex_GP'].values
            hour_series = df_seg_id['hour'].values
            for t in range(0,len(TI_series)-(self.window+self.horizon)):
                x = TI_series[t:t+self.window]
                y = TI_series[t+self.window:(t+self.window+self.horizon)]
                h = hour_series[t:t+self.window]
                x_cat = np.dstack([x, h])
                self.inputs.append(x_cat) #X_train or X_test
                self.targets.append(y) #y_train or y_test
         
    def __len__(self):
        return len(self.inputs)


    def __getitem__(self,idx):
        X = torch.tensor(self.inputs[idx],dtype=torch.float32).reshape(self.window, 2)
        y=torch.tensor(self.targets[idx],dtype=torch.float32)

        return {'inputs':X,'outputs':y}
    

def train_val(model, epochs, dataloader_train, dataloader_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the selected device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mean_loss = []
    mean_test_loss = []
    n_epochs = epochs

    for it in tqdm(range(n_epochs)):
        losses = []
        for i_batch, sample_batched in enumerate(dataloader_train):
            optimizer.zero_grad()
            inputs = sample_batched['inputs'].to(device) 
            outputs = model(inputs)
            targets = sample_batched['outputs'].to(device)  
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        test_losses = []
        for i_test, sample_test in enumerate(dataloader_test):
            optimizer.zero_grad()
            inputs = sample_test['inputs'].to(device)  
            outputs = model(inputs)
            targets = sample_test['outputs'].to(device)  
            test_loss = criterion(outputs, targets)
            test_losses.append(test_loss.item())
        mean_loss.append(np.mean(losses))
        mean_test_loss.append(np.mean(test_losses))

        print(f'Epoch {it+1}/{n_epochs}, Training Loss: {np.mean(losses):.4f}, Testing Loss: {np.mean(test_losses):.4f}')
   

    plt.plot(mean_loss,'k',label='training')
    plt.plot(mean_test_loss,'m',label='testing')
    plt.legend()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def evaluation(model, dataloader_test, device):
    maes = []
    mapes = []
    model.eval()
    with torch.no_grad():
        for i_test, sample_test in enumerate(dataloader_test):
            inputs = sample_test['inputs'].to(device)
            targets = sample_test['outputs'].to(device)
            preds = model(inputs)
            mae = mean_absolute_error(targets.cpu().numpy(), preds.cpu().numpy())
            mape = mean_absolute_percentage_error(targets.cpu().numpy(), preds.cpu().numpy())
            maes.append(mae)
            mapes.append(mape)
    return {'mae': np.mean(maes), 'mape': np.mean(mapes)}


class Predict(Dataset):
    def __init__(self, window, horizon):
        self.window = window
        self.horizon=horizon
        self.inputs = []
        test_pkls = os.listdir('Test')

        for test_pkl in test_pkls:
            df = pd.read_pickle(os.path.join('Test', test_pkl))
            df['time'] = pd.to_datetime(df['time'])
            df['unix_timestamp'] = df['time'].astype(int) / 10**9
            df = df.sort_values(by='time', ascending=True)
            seg_ids = df['segmentID'].unique()                
            for segid in seg_ids:
                df_seg_id = df[df['segmentID'] == segid]
                df_seg_id = df_seg_id.fillna(method="ffill")
                df_seg_id = df_seg_id.sort_values(by='time', ascending=True)
                df_seg_id['hour'] = df_seg_id['time'].dt.hour
                df_seg_id['min'] = df_seg_id['time'].dt.minute
                df_seg_id['dow'] = df_seg_id['time'].dt.weekday
                TI_series = df_seg_id['TrafficIndex_GP'].values
                hour_series = df_seg_id['hour'].values
                min_series = df_seg_id['min'].values
                day_series = df_seg_id['dow'].values
                for t in range(0,1):
                    x = TI_series[t:t+self.window]
                    h = hour_series[t:t+self.window]
                    m = min_series[t:t+self.window]
                    d = day_series[t:t+self.window]
                    x_cat = np.dstack([x, h])
                    self.inputs.append(x_cat)
        print(len(self.inputs))
                    
    def __len__(self):
        return len(self.inputs)


    def __getitem__(self,idx):
        X = torch.tensor(self.inputs[idx],dtype=torch.float32).reshape(self.window, 2)
        return {'inputs':X}
    
def predict_json(model, model_name, model_weight,dataloader_pred, device ):
    model = model
    model.to(device)
    model.load_state_dict(torch.load(model_weight))

    with torch.no_grad():
        y=[]
        for i_test, sample_test in enumerate(dataloader_pred):
            outputs = model(sample_test['inputs'].to(device))
            y_pred = outputs.cpu().detach().numpy()
            y.append(y_pred)

    pkl_path='tps_df.pkl'
    df = pd.read_pickle(pkl_path)
    seg_ids = df['segmentID'].unique()
    seg_ids=[str(i) for i in seg_ids]

    for i in y:
        df=pd.DataFrame(i).transpose()
        df.columns=seg_ids
    # First, make sure the prediction result is a martrix with shape (12, 87).
    output=[]
    for i in range(len(y)):
        D=pd.DataFrame(y[i]).transpose()
        D.columns=seg_ids
        output.append(D)

    # # Convert the datetime to unix format as follows. Please change the range the datetime correctly!
    start_hour = 6
    start_day = 2
    end_hour = 9
    end_day = 2
    for i in range(0, 15):
        start_time = f'2020-06-{start_day:02d} {start_hour:02d}:15:00'
        end_time = f'2020-06-{end_day:02d} {end_hour:02d}:00:00'  
        output[i].index = pd.date_range(start=start_time, end=end_time, freq='15min').astype(int) / 10**9
        start_hour+=1
        end_hour+=1
        start_day +=1
        end_day +=1

    prediction_result=pd.DataFrame()
    for i in output:
        prediction_result=pd.concat([prediction_result,i])
    
    final_output = prediction_result
    final_output = final_output.round(3)
    final_output.to_json(f'{model_name}_prediction_result.json', double_precision=3)

    return final_output