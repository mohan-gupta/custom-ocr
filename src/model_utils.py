import torch

import config

def save_model(epoch, model, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, config.MODEL_PATH)

def load_model():
    chkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    return chkpoint