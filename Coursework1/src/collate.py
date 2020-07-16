import torch

def qc_collate_fn_bow(QCDataset):
    data,label = [],[]
    for dataset in QCDataset:
        data.append(dataset[0])
        label.append(dataset[1])
    return data,label

def qc_collate_fn_bilstm(QCDataset):
    length,data,label = [],[],[]
    for dataset in QCDataset:
        data.append(dataset[0])
        label.append(dataset[1])
        length.append(len(dataset[0]))
    data = torch.nn.utils.rnn.pad_sequence(data,padding_value=0)
    return data,label,length