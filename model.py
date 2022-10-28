import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.resnet = models.resnet18(weights=None)
        cnn_layers = list(self.resnet.children())[:-1]
        self.cnn = nn.Sequential(*cnn_layers)
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc3 = nn.Linear(512, 300)

    def forward(self, x_3d):
        cnn_output_list = []
        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.fc3(x)

            cnn_output_list.append(x)
            
        cnn_embedding_out = torch.stack(cnn_output_list, dim=0).transpose(0, 1)
       
        
        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self):
        super(RNNDecoder, self).__init__()

        self.num_classes = 39

        self.rnn = nn.GRU(input_size=300, hidden_size=256, num_layers=3, batch_first=True)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)

        x = self.fc1(rnn_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)

        return x