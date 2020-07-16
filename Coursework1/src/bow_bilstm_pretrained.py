from torch import nn
import torch
from ffnn import FFNN

class BOW_BiLSTM_PRE(nn.Module):

    def __init__(self, embeddings, bilstm_hidden_size, ffnn_hidden_size, ffnn_output_size, freeze = True):
        super(BOW_BiLSTM_PRE, self).__init__()
        # Get the dimension of embeddings as the input size of bilstm
        _, self.input_size = embeddings.size()
        self.bilstm_hidden_size = bilstm_hidden_size
        self.ffnn_hidden_size = ffnn_hidden_size
        self.ffnn_output_size = ffnn_output_size
        self.freeze = freeze
        # Use pretrained embeddings, bag of words layer
        self.bow = nn.EmbeddingBag.\
                from_pretrained(
                    embeddings,
                    freeze=self.freeze,
                    mode='mean'
                )
        # Bilstm network
        self.bilstm = nn.LSTM(self.input_size,self.bilstm_hidden_size, bidirectional=True)
        # Feed forward neural network with one hidden layer 
        self.ffnn = FFNN(self.bilstm_hidden_size*2,self.ffnn_hidden_size, self.ffnn_output_size)
        # Softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, lengths):
        # Remove the paddings in the data
        input = input.transpose(0,1)
        batch = list()
        offsets = list()
        offset = 0
        for i in range(input.size()[0]):
            batch += input[i][:lengths[i]]
            offsets.append(offset)
            offset += lengths[i]
        # Data --> BOW
        vec = self.bow(torch.tensor(batch), torch.tensor(offsets))
        # BOW --> BiLSTM
        lstm_out, _ = self.bilstm(vec.view(1, -1, self.input_size))
        # BiLSTM --> FFNN
        ffnn = self.ffnn(lstm_out.view(-1, self.bilstm_hidden_size*2))
        # FFNN --> log softmax
        return self.log_softmax(ffnn)
