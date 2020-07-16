import torch
from torch import nn
from ffnn import FFNN

class BiLSTM_FFNN_PRE(nn.Module):
    def __init__(self, embeddings, bilstm_hidden_size, ffnn_hidden_size, ffnn_output_size, freeze = True):
        super(BiLSTM_FFNN_PRE, self).__init__()
        # Get the dimension of embeddings as the input size of bilstm
        _, self.input_size = embeddings.size()
        self.bilstm_hidden_size = bilstm_hidden_size
        self.ffnn_hidden_size = ffnn_hidden_size
        self.ffnn_output_size = ffnn_output_size
        self.freeze = freeze

        # Use pretrained embeddings
        self.embeddingLayer = nn.Embedding.from_pretrained(embeddings, self.freeze)
        
        # Bilstm network
        self.bilstm = nn.LSTM(self.input_size,self.bilstm_hidden_size, bidirectional=True)
        # Feed forward neural network with one hidden layer 
        self.ffnn = FFNN(self.bilstm_hidden_size*2,self.ffnn_hidden_size, self.ffnn_output_size)
        # Softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, lengths):
        # Get the embeddings of the padded data
        sequence_embed = self.embeddingLayer(input)
        # Pack the embeddings of padded data
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            sequence_embed,
            lengths, enforce_sorted=False
        )
        # Packed embeddings --> BiLSTM --> still packed output
        packed_output,_ = self.bilstm(pack)
        # Packed output of BiLSTM --> Re-pad
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        # Get the actual mean of re-padded vectors (without padded values)
        for i in range(output.size()[1]):
            if i==0:
                vec = output.index_select(1,torch.tensor(i))\
                    .squeeze(1)\
                        .index_select(0,torch.tensor(lengths[i]-1))\
                            .mean(dim=0)\
                                .unsqueeze(0)
            else:
                vec = torch.cat((vec,output.index_select(1,torch.tensor(i)).squeeze(1).index_select(0,torch.tensor(lengths[i]-1)).mean(dim=0).unsqueeze(0)),0)
        # BiLSTM --> FFNN
        ffnn = self.ffnn(vec)
        # FFNN --> log softmax
        return self.log_softmax(ffnn)