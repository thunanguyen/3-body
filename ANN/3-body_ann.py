import sys
import os
import torch
import argparse
from model import ANN
from dataloader import Data
from torch.utils.data import DataLoader




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Config file")
    parser.add_argument('--num_epochs', type=int, default=15000, help='The total number of epochs used for training')
    parser.add_argument('--epoch', type=int, default=0, help='The starting epoch used for loading the last checkpoint epoch')
    parser.add_argument('--saved_every_epoch', type=int, default=1000, help='The number of epochs for checkpoint')
    parser.add_argument('--batch_size', type=int, default=1000, help='The batch size used for training')
    parser.add_argument('--num_workers', type=int, default=8, help='The numbers of processors used for loading the data')
    
    args = parser.parse_args()
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('saved_models', exist_ok=True)

    data = DataLoader( Data("./data/training"),
                       batch_size=args.batch_size,
                       shuffle=True,
                       num_workers=args.num_workers)
    
    model = ANN()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0002,
                                 betas=(0.9, 0.999))

    if args.epoch > 0:
        model.load_state_dict(torch.load('saved_models/model_%d.pth' % args.epoch))
    
    for epoch in range(args.epoch, args.num_epochs):
        for i, items in enumerate(data):
            init = items['init']
            init_pos, init_vel, init_t = init['pos'], init['vel'], init['t']
            init_t = init_t.view(-1,1)

            final = items['final']
            final_pos, final_vel = final['pos'], final['vel']

            optimizer.zero_grad()
            
            #pred_pos, pred_vel = model(init_pos.float(), init_vel.float(), init_t.float())
            pred_pos, pred_vel = model(init_pos.float(), init_vel.float(), init_t.float())
            loss = (criterion(pred_pos, final_pos) + criterion(pred_vel, final_vel)) / 2

            loss.backward()
            optimizer.step()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %.4f]"
                % ( epoch,
                    args.num_epochs,
                    i,
                    len(data),
                    loss.item())
            )

        if epoch % args.saved_every_epoch == 0:
            torch.save(model.state_dict(), "saved_models/model_%d.pth" % epoch)
            
