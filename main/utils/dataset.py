import torch
import numpy as np

class IFEDataset(torch.utils.data.Dataset):
    def __init__(self, x, label, transform=None):
        self.x = x
        self.label = label
        self.transform = transform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        data_x = torch.from_numpy(self.x[idx]).type(torch.FloatTensor)
        data_y = torch.from_numpy(np.array(self.label[idx])).type(torch.long)
        return data_x, data_y