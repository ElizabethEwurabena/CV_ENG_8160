import random,math
import torch.nn as nn
import numpy as np
import os,torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
class MyRNNCell(nn.Module):
    def __init__(self,input_size, hidden_size, bias=True):
        super(MyRNNCell,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias 
        self.x2h = nn.Linear(input_size,hidden_size, bias=self.bias)
        self.h2h = nn.Linear(hidden_size,hidden_size,bias=self.bias)
        self.nonlinearity = nn.ReLU()

    def forward(self,x):
        h0 = torch.zeros(x.size(0), self.hidden_size).to(device)# added to device here
        hy = (self.x2h(x) + self.h2h(h0))
        hy = self.nonlinearity(hy)
        return hy 

class MyRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,output_size, bias=True):
        super(MyRNN,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias 
        self.output_size = output_size
        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(MyRNNCell(self.input_size,self.hidden_size))
        for layer in range(0,self.num_layers):
            self.rnn_cell_list.append(MyRNNCell(self.hidden_size,self.hidden_size))

        self.fc = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x):
        ## push to cuda if available
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)
        outputs = []
        hidden = list()

        for layer in range(self.num_layers):
            hidden.append(h0[layer,:,:])

        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_layer = self.rnn_cell_list[layer](x[:, t, :])
                else:
                    hidden_layer = self.rnn_cell_list[layer](hidden[layer-1])
                hidden[layer] = hidden_layer
            outputs.append(hidden_layer)
        output = outputs[-1].squeeze()
        output = self.fc(output)
        # output = output.reshape(:,:,-1)

        return output 

class MyGRU(nn.Module):
  # build a constructor to define the inputs for the rnn module
  def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
    super(MyGRU,self).__init__()
    self.D = n_inputs
    self.M = n_hidden
    self.K = n_outputs
    self.L = n_rnnlayers
    self.relu = nn.ReLU()

    #batch_first means reshaping input to N x T x D
    self.rnn = nn.GRU(
        input_size=self.D,
        hidden_size=self.M,
        num_layers=self.L,
        # nonlinearity='relu',
        batch_first= True)
    
    self.fc = nn.Linear(self.M, self.K)
    
  def forward(self, X):
    # initialize hidden states
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    # out is N x T x M
    # second output returns hidden state at each hidden layers
    out,_ = self.rnn(X,h0)

    # we only want the hidden state at the final step
    # out is now N x M
    out = self.fc(self.relu(out[:,-1,:]))
    # out = self.fc(out[:,-1,:])

    return out

class MyLSTM(nn.Module):
  # build a constructor to define the inputs for the rnn module
  def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
    super(MyLSTM,self).__init__()
    self.D = n_inputs
    self.M = n_hidden
    self.K = n_outputs
    self.L = n_rnnlayers
    # self.embedding_layer = nn.Embedding(23, 1)
    self.relu = nn.ReLU()

    #batch_first means reshaping input to N x T x D
    self.rnn = nn.LSTM(
        input_size=self.D,
        hidden_size=self.M,
        num_layers=self.L,
        batch_first= True)
    
    self.fc = nn.Linear(self.M, self.K)
    
  def forward(self, X):
    # initialize hidden states
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    out,_ = self.rnn(X,(h0,c0))

    # we only want the hidden state at the final step
    # out is now N x M
    out = self.fc(out[:,-1,:])

    return out
