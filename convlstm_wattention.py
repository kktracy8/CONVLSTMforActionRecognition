import torch
import torch.nn as nn
import os
import numpy as np
from ConvLSTMcell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, attention):
        """
        ConvLSTM with optional temporal attention.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding for the convolution.
            attention (bool): Whether to apply temporal attention.
        """
        super(ConvLSTM, self).__init__()
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = attention
        self.out_channels = out_channels

        if self.attention:
            self.temporal_attention_fc = nn.Linear(out_channels, 1)  # Attention mechanism

        self.hook_data = {
            'input': [],
            #'output': [],
            'grad_input': [],
            'grad_output': []
        }
        self.hooks = []

    def temporal_attention(self, hidden_states):
        """
        Applies temporal attention to the hidden states.

        Args:
            hidden_states (Tensor): Hidden states of shape (batch_size, seq_len, out_channels, height, width).

        Returns:
            Tensor: Attention-weighted output of shape (batch_size, out_channels, height, width).
        """
        batch_size, seq_len, out_channels, height, width = hidden_states.size()
        
        # Global pooling to reduce spatial dimensions
        hidden_flat = hidden_states.view(batch_size, seq_len, out_channels, -1).mean(dim=-1)  # (batch_size, seq_len, out_channels)
        
        # Compute attention scores
        attention_scores = self.temporal_attention_fc(hidden_flat).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, seq_len, 1, 1, 1)
        
        # Apply attention weights
        weighted_hidden = hidden_states * attention_weights  # (batch_size, seq_len, out_channels, height, width)
        output = weighted_hidden.sum(dim=1)  # (batch_size, out_channels, height, width)
        return output

    def forward_hook(self, module, input, output):
        self.hook_data['input'].append(input[0].cpu().detach().numpy()) 

    def backward_hook(self, module, grad_input, grad_output):
        self.hook_data['grad_input'].append(grad_input[0].cpu().detach().numpy()) 
        self.hook_data['grad_output'].append(grad_output[0].cpu().detach().numpy())

    def register_hooks_for_frame(self, idx, frame_idx):
        if frame_idx is not None and idx == frame_idx:
            print(f"Registering hooks for frame {frame_idx}")
            forward_hook = self.convLSTMcell.register_forward_hook(self.forward_hook)
            backward_hook = self.convLSTMcell.register_backward_hook(self.backward_hook)
            self.hooks.append((forward_hook, backward_hook))

    def clear_hooks(self):
        for forward_hook, backward_hook in self.hooks:
            forward_hook.remove()
            backward_hook.remove()
        self.hooks = []

    def forward(self, input, register_hooks=False, frame_idx=None):
        """
        Forward pass for ConvLSTM with temporal attention.

        Args:
            input (Tensor): Input sequence of shape (batch_size, seq_len, in_channels, height, width).

        Returns:
            Tensor: Output with temporal attention applied (if enabled) or the last hidden state.
        """
        input = input.float()
        batch_size, seq_len, _, height, width = input.size()

        # Initialize hidden states
        hidden_states = torch.zeros(batch_size, seq_len, self.out_channels, height, width, device=device)
        hidden = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        cell = torch.zeros(batch_size, self.out_channels, height, width, device=device)

        if register_hooks:
            for t in range(seq_len):
                self.register_hooks_for_frame(t, frame_idx)

        for t in range(seq_len):
            hidden, cell = self.convLSTMcell(input[:, t, :, :, :], hidden, cell)
            hidden_states[:, t, :, :, :] = hidden
        del cell

        # Apply attention if enabled
        if self.attention:
            output = self.temporal_attention(hidden_states)
        else:
            output = hidden_states#[:, -1, :, :, :]  # Last hidden state if no attention

        #if register_hooks:
        #if register_hooks and len(self.hook_data['grad_output']) > 0:
        #    np.save('hook_data_grad_output.npy', np.array(self.hook_data['grad_output']))
        #    self.clear_hooks()

        return output
