import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights


class CNNEncoder(nn.Module):
    def __init__(self):
      super().__init__()
      self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
      self.resnet.fc = nn.Identity() 
    def forward(self, x:torch.Tensor):
        with torch.no_grad():
            features = self.resnet(x)
        return features

    

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,encoder):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers    
        self.img_projector = nn.Linear(2048, hidden_size)
        self.encoder = encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x,features):
        features = self.encoder(features)
        features = self.img_projector(x)
        h0 = c0 = features
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
