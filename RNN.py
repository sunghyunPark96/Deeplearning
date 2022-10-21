import tqdm
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

class Netflix(Dataset):
    def __init__(self,data):
        super(Netflix,self).__init__()
        self.csv = data
        self.data = self.csv.iloc[:,1:4]
        
        for key in self.data.columns:
            value_max = self.data[key].max()
            value_min = self.data[key].min()
            self.data[key] = (self.data[key]- value_min) / (value_max-value_min)

        self.label = self.csv.iloc[:,5]
        self.label = ((self.label - self.label.min())/(self.label.max()-self.label.min()))

    def __len__(self):

        return len(self.label) -30

    def __getitem__(self,index):
        data = torch.tensor(self.data.iloc[index:index+30].values,dtype=torch.float32)
        label = torch.tensor(self.label.iloc[index+30],dtype=torch.float32)

        return data,label

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size=3,hidden_size=8,num_layers=5,batch_first=True)
        self.fc1 = nn.Linear(in_features=8*30,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=1)
        self.relu = nn.ReLU()

    def forward(self,x,h0):
        x, hn = self.rnn(x,h0)
        x = torch.reshape(x, (x.shape[0],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.flatten(x)

        return x

if __name__ == "__main__":
    data = pd.read_csv("C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 알고리즘 구현/RNN/NFLX.csv")

    #EDA 
    print(data.head(5))
    data_EDA = data.iloc[:,1:5] 
    # hist = data_EDA.hist()
    # plt.show()

    #DATASET MAKE
    dataset = Netflix(data)
    train_dataloader = DataLoader(dataset=dataset, batch_size=32)
    test_dataloader = DataLoader(dataset=dataset, batch_size=1)

    #Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")   

    #Model
    RNN_model = RNN().to(device=device)
    RNN_test_model = RNN().to(device=device)

    #Learning
    lr = 0.0001
    epoch = 200
    save_path = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 알고리즘 구현/RNN/model/RNN_epoch200.pt"
    criterion = nn.MSELoss()
    optim = Adam(params=RNN_model.parameters(),lr=lr)

    signal = input(str("train : y or test : n "))

    if signal == "y":
        for i in range(epoch):
            RNN_model.train()
            iterator = tqdm.tqdm(train_dataloader)
            epoch_loss = 0

            for data,label in iterator:
                optim.zero_grad()
                h0 = torch.zeros(5, data.shape[0],8).to(device=device) # h0 초기화 (5,30,3,8)

                pred = RNN_model(data.to(device=device),h0)
                loss = criterion(pred,label.to(device=device))
                batch_loss = loss.item()
                epoch_loss += batch_loss

                loss.backward()
                optim.step()

            avg_epoch_loss = epoch_loss / len(train_dataloader)

            iterator.set_description(f"epoch{i+1} loss : {avg_epoch_loss}")
            print(iterator)

        torch.save(RNN_model.state_dict(),save_path)

    if signal == "n":

        preds = []
        total_loss = 0

        with torch.no_grad():
            RNN_test_model.eval()
            RNN_test_model.load_state_dict(torch.load(save_path))

            for data,label in test_dataloader:
                h0 = torch.zeros(5,data.shape[0],8).to(device=device)
                pred = RNN_test_model(data.to(device=device),h0)
                preds.append(pred.item())

                loss = criterion(pred, label.to(device=device))

                total_loss += loss/(len(test_dataloader))

        plt.plot(preds, label="prediction")
        plt.plot(dataset.label[30:].reset_index().drop(columns="index"),label="actual")
        plt.legend()
        plt.show()