from torch.utils.data import Dataset

class QCDataset(Dataset):
    def __init__(self,xy):
        self.data,self.label = xy

    def __getitem__(self, index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)