from convlstm_wattention import ConvLSTM
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, device, attention, hidden_dim=64):
        super(Model, self).__init__()
        self.conv_lstm = ConvLSTM(input_size, hidden_dim, kernel_size, stride, padding, attention).to(device)
        self.fc = nn.Linear(hidden_dim, output_size).to(device)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, inputs,save_hooks=False):
        if inputs.dim() != 5:  # Expecting (batch, seq_len, channels, height, width)
            raise ValueError(f"Expected shape (batch, seq_len, channels, height, width), got {inputs.shape}")
        frame_idx = int(inputs.shape[1]/2)
        if save_hooks:
            lstm_output = self.conv_lstm(inputs, register_hooks=True, frame_idx=frame_idx)  # (batch_size, hidden_dim, H, W)
        else:
            lstm_output = self.conv_lstm(inputs, register_hooks=False, frame_idx=frame_idx)  # (batch_size, hidden_dim, H, W)        
        lstm_output = lstm_output.mean(dim=[3, 4])  # Global average pooling
        output = self.fc(lstm_output)  # (batch_size, output_size)
        return self.soft(output)
