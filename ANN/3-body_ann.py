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
                    num_epochs,
                    i,
                    len(data),
                    loss.item())
            )

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), "saved_models/model_%d.pth" % epoch)
            
