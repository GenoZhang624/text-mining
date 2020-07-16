import torch
from torch import nn
from ffnn import FFNN

class BOW_FFNN_PRE(nn.Module):
    def __init__(self, embeddings, hidden_size, output_size, freeze = True):
        super(BOW_FFNN_PRE,self).__init__()
        self.voca_size,self.input_size = embeddings.size()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.freeze = freeze
        # BOW layer
        self.bow = nn.EmbeddingBag.from_pretrained(embeddings,self.freeze,mode='mean')
        # FFNN with single hidden layer
        self.ffnn = FFNN(self.input_size,self.hidden_size,self.output_size)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input):
        # batch: a very long list storing all the sentences' representations
        # offsets: a list of the beginning of each sentence
        batch = list()
        offsets = list()
        offset = 0
        for i in range(len(input)):
            batch += input[i] # .extend()
            offsets.append(offset)
            offset += len(input[i])

        # Data --> BOW
        vec = self.bow(torch.tensor(batch),torch.tensor(offsets))
        # BOW --> FFNN
        ffnn = self.ffnn(vec)
        # FFNN --> log softmax
        return self.log_softmax(ffnn)
