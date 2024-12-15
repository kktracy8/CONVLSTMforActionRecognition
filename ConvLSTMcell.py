import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        ConvLSTM cell implementation.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Kernel size for convolutions.
            stride (int): Stride for convolutions.
            padding (int): Padding for convolutions.
        """
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (1,1)
        self.padding = (1,1)

        # Input gate
        self.wii = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bi = nn.Parameter(torch.zeros(out_channels))
        self.whi = nn.Parameter(torch.zeros(out_channels, out_channels, kernel_size[0], kernel_size[1]))

        # Forget gate
        self.wif = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bf = nn.Parameter(torch.zeros(out_channels))
        self.whf = nn.Parameter(torch.zeros(out_channels, out_channels, kernel_size[0], kernel_size[1]))

        # Cell gate
        self.wig = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bg = nn.Parameter(torch.zeros(out_channels))
        self.whg = nn.Parameter(torch.zeros(out_channels, out_channels, kernel_size[0], kernel_size[1]))

        # Output gate
        self.wio = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bo = nn.Parameter(torch.zeros(out_channels))
        self.who = nn.Parameter(torch.zeros(out_channels, out_channels, kernel_size[0], kernel_size[1]))

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize weights for ConvLSTMCell using Xavier initialization.
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def conv2d(self, input, w_input, hidden, w_hidden, b):
        ### input shape: N,in_channel,iH,iW
        ### input weight kernel shape: out_channel,in_channel,kH,kW
        ### hidden shape: N,out_channel,iH,iW
        ### hidden weight kernel shape: out_channel,out_channel,kH,kW
        ### bias shape: out_channel -> need to broadcast
        input_conv = nn.functional.conv2d(input, w_input, stride=1, padding='same')
        h_conv = nn.functional.conv2d(hidden, w_hidden, stride=1, padding='same')
        gate_data = input_conv + h_conv + b.view(1,self.out_channels,1,1)
        return gate_data

    def forward(self, input, hidden, cell):
        i_t = torch.sigmoid(self.conv2d(input, self.wii, hidden, self.whi, self.bi))
        f_t = torch.sigmoid(self.conv2d(input, self.wif, hidden, self.whf, self.bf))
        o_t = torch.sigmoid(self.conv2d(input, self.wio, hidden, self.who, self.bo))
        g_t = torch.tanh(self.conv2d(input, self.wig, hidden, self.whg, self.bg))
        c_t = f_t*cell + i_t*g_t
        h_t = o_t*torch.tanh(c_t)
        return h_t,c_t