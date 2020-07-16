import torch

class FFNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(FFNN,self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        fc1 = self.fc1(inp)
        relu1 = self.relu1(fc1)
        fc2 = self.fc2(relu1)
        return fc2