import torch
import torch.nn as nn
from ConvLSTMcell import ConvLSTMCell

class BaseCases():
    def __init__(self,name):
        self.name = name

    def run_test(self):
        ### To be defined at individual subclass methods
        pass

    def assertion_equal(self,actual,expected,equal_function):
        self.actual = actual
        self.expected = expected
        self.equal_function = equal_function
        assert self.expected == self.actual, f"Assertion for {self.name} has FAILED! Expected {self.expected} but got {self.actual} for {self.equal_function}"
    
    def assertion_bool(self,bool_val,bool_function):
        self.bool_val = bool_val
        self.bool_function = bool_function
        assert self.bool_val, f"Assertion for {self.name} has FAILED for {self.bool_function}!"

class Test_ConvLSTMCell(BaseCases):
    def __init__(self,batch_size,input_channels,output_channels,height,width,kernel_size,stride,padding):
        super().__init__("Test ConvLSTMCell Function")
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.convlstmcell = ConvLSTMCell(self.output_channels,self.kernel_size,self.stride,self.padding)

    def run_test(self):
        input_tensor = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        hidden_state = torch.zeros(self.batch_size, self.output_channels, self.height, self.width)
        cell_state = torch.zeros(self.batch_size, self.output_channels, self.height, self.width)
        hidden_state, cell_state = self.convlstmcell.forward(input_tensor, hidden_state, cell_state)
        expected_shape = (self.batch_size, self.output_channels, self.height, self.width)
        super().assertion_equal(hidden_state.shape,expected_shape,"hidden state shape")
        super().assertion_equal(cell_state.shape,expected_shape,"cell state shape")
        print("Test ConvLSTMCell Function passed!")

batch_size = 2
input_channels = 1
output_channels = 4
height = 128
width = 128
kernel_size = 3
stride = 1
padding = 1

print("Running ConvLSTMCell Shape test...")
test_convlstm_cell = Test_ConvLSTMCell(batch_size,input_channels,output_channels,height,width,kernel_size,stride,padding)
test_convlstm_cell.run_test()



