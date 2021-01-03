import os
import sys
import math
import torch
import argparse
import numpy as np
from model import ANN
from dataloader import Data
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Config file")
    parser.add_argument('--model', type=str, default='saved_models/model_final.pth', help='The saved model used for testing')
    parser.add_argument('--num_workers', type=int, default=8, help='The numbers of processors used for loading the data')
    
    args = parser.parse_args()
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = DataLoader( Data("./data/testing"),
                           batch_size=1,
                           shuffle=True,
                           num_workers=args.num_workers)
        
    model = ANN()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    
    loss_mse = []
    loss_mae = []

    for i, items in enumerate(data):
        init = items['init']
        init_pos, init_vel, init_t = init['pos'], init['vel'], init['t']
        init_t = init_t.view(-1,1)

        final = items['final']
        final_pos, final_vel = final['pos'], final['vel']
        pred_pos, pred_vel = model(init_pos.float(), init_vel.float(), init_t.float())
        
        loss_mse.append(((mse(pred_pos, final_pos) + mse(pred_vel, final_vel)) / 2).item())
        loss_mae.append(((mae(pred_pos, final_pos) + mae(pred_vel, final_vel)) / 2).item())


    loss_mse = np.array(loss_mse)
    loss_mae = np.array(loss_mae)

    print("The average MAE error for the testing dataset is", loss_mae.mean())
    print("The average RMSE error for the testing dataset is", math.sqrt(loss_mse.mean()))
