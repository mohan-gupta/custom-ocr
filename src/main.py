import os
import pandas as pd

import torch
import torch.nn as nn

import config
from dataset import get_loaders
from model import TextRecognizer
from engine import train, evaluate
from label_processor import encode_labels
from model_utils import save_model, load_model

def run_training(load=False):
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, "train/img_rcg.csv"), keep_default_na=False)
    valid_df = pd.read_csv(os.path.join(config.DATA_PATH, "valid/img_rcg.csv"), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, "test/img_rcg.csv"), keep_default_na=False)

    labels = encode_labels(train_df['label'], valid_df['label'], test_df['label'], load)
    
    train_loader = get_loaders(
        image_paths=train_df['img_name'],
        labels=labels['train_labels'],
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        split="train",
        max_len=config.MAX_LEN,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
        )
    
    valid_loader = get_loaders(
        image_paths=valid_df['img_name'],
        labels=labels['valid_labels'],
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        split="valid",
        max_len=config.MAX_LEN,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True
        )
    
    test_loader = get_loaders(
        image_paths=test_df['img_name'],
        labels=labels['test_labels'],
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        split="test",
        max_len=config.MAX_LEN,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=True
        )
    
    # add 1 to num classes as 0 is preserved for adding and epsilon token.
    model = TextRecognizer(labels['num_classes']+1)
    model.to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.DR)

    if load is True:
        chkpoint = load_model()
        start_epoch = chkpoint["epoch"]

        model.load_state_dict(chkpoint['model_state_dict'])
        optimizer.load_state_dict(chkpoint['optimizer_state_dict'])
        print("Loaded the model and optimizer")
        print(f"Training Resumed after epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Training Started")
    
    loss_fn = nn.CTCLoss(blank=config.PAD_VAL, zero_infinity=True)
    # Sometimes loss first becomes inf before NaN, in which case the inf loss (and gradients) can be reset to zero.
    
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(start_epoch, start_epoch+config.EPOCHS+1):
        print(f"Epoch: {epoch}")
        train_loss = train(model, train_loader, loss_fn, optimizer)
        val_preds, val_loss = evaluate(model, valid_loader, loss_fn)
        print(f"Training Loss = {train_loss}, Validation Loss = {val_loss}")

        if epoch % 10 == 0:
            save_model(epoch, model, optimizer, val_loss)
    
    test_preds, test_loss = evaluate(model, test_loader, loss_fn)
    print(f"Test Loss = {test_loss}")

if __name__ == "__main__":
    run_training(load=False)