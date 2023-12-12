import torch
import torch.nn as nn
from torch.nn import functional as F

class TextRecognizer(nn.Module):
    def __init__(self, num_chars) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(64, 128, (3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.conv5 = nn.Conv2d(128, 256, (3,3), padding=(1,1))
        self.conv6 = nn.Conv2d(256, 256, (3,3), padding=(1,1))
        self.conv7 = nn.Conv2d(256, 64, (2,2), padding=(1,1))
        
        self.maxpool2 = nn.MaxPool2d((2,2))
        self.maxpool4 = nn.MaxPool2d((2,1))
        self.maxpool5 = nn.MaxPool2d((2,1))
        self.maxpool6 = nn.MaxPool2d((2,1))
        
        self.bnorm2 = nn.BatchNorm2d(64)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.bnorm4 = nn.BatchNorm2d(128)
        self.bnorm5 = nn.BatchNorm2d(256)
        self.bnorm6 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(
            input_size=320, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1
            )
        
        self.linear = nn.Linear(256, num_chars+1)
        
    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        
        x = F.selu(self.conv1(images))
        
        x = F.selu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.maxpool2(x)
        
        x = F.selu(self.conv3(x))
        x = self.bnorm3(x)
        
        x = F.selu(self.conv4(x))
        x = self.bnorm4(x)
        x = self.maxpool4(x)
        
        x = F.selu(self.conv5(x))
        x = self.bnorm5(x)
        x = self.maxpool5(x)
        
        x = F.selu(self.conv6(x))
        x = self.bnorm6(x)
        x = self.maxpool6(x)
        
        x = F.selu(self.conv7(x))
        
        x = x.permute((0, 3, 1, 2))
        
        x = x.view((bs, x.size(1), -1))
        x = self.dropout(x)
        
        seq_out, _ = self.lstm(x)
        
        output = self.linear(seq_out)
        
        return output

if __name__ == "__main__":
    image = torch.rand((2,3,75,300))
    model = TextRecognizer(96)
    _  = model(image) # output size: (batch_size, max_len, num_classes)