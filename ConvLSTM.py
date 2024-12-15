import torch
import torch.nn as nn
from ConvLSTMcell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTM(nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        self.convLSTMcell = ConvLSTMCell(1,out_channels, kernel_size, stride, padding)

    def forward(self, input):
        ### input shape: N,T,in_channel,iH,iW
        batch_size, seq_len, _, height, width = input.size()
        output = torch.zeros(batch_size, seq_len, self.out_channels, height, width, device=device)
        hidden = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        cell = torch.zeros(batch_size,self.out_channels, height, width, device=device)

        for t in range(seq_len):
            hidden, cell = self.convLSTMcell.forward(input[:,t,:,:,:], hidden, cell)
            output[:,t,:,:,:] = hidden

        return output
