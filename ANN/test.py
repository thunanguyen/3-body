from dataloader import Data
from torch.utils.data import DataLoader
from model import ANN
import torch
import sys



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoader( Data("./data"),
                       batch_size=1000,
                       shuffle=True,
                       num_workers=4)
    
    model = ANN()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0002,
                                 betas=(0.9, 0.999))

    num_epochs = 90000
    
    for epoch in range(num_epochs):
        for i, items in enumerate(data):
            print()
