from tqdm import tqdm

import torch
from torch.nn import functional as F

import config

def train_one_step(model, data, loss_fn, optimizer):
    optimizer.zero_grad()

    for k,v in data.items():
        data[k] = v.to(config.DEVICE)

    preds = model(data['image'])
    
    batch_size, seq_len, _ = preds.size()
    
    preds_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long)
    
    # the input for CTC loss is (seq_len, batch_size, embedding_size)
    preds = preds.permute((1,0,2))
    # We need to pass the log softmax values as input to CTC loss
    loss_inp = F.log_softmax(preds, dim=2)
    
    loss = loss_fn(loss_inp, data['label'], preds_lengths, data['label_len'])

    loss.backward()
    optimizer.step()

    return loss

def train(model, dataloader, loss_fn, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 5 #len(dataloader)
    loop = tqdm(enumerate(dataloader), total=num_batches)

    for batch_idx, data in loop:
        loss = train_one_step(model, data, loss_fn, optimizer)

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        loop.set_postfix({'loss': loss.item()})
        
        if batch_idx==4:
            break

    train_loss = total_loss/num_batches

    return train_loss

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=num_batches)

    final_preds = []
    with torch.no_grad():
      for batch_idx, data in loop:
            for k,v in data.items():
                data[k] = v.to(config.DEVICE)

            preds = model(data['image'])
          
            batch_size, seq_len, _ = preds.size()
    
            preds_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.int32)
            
            # the input for CTC loss is (seq_len, batch_size, embedding_size)
            preds = preds.permute((1,0,2))
            # We need to pass the log softmax values as input to CTC loss
            loss_inp = F.log_softmax(preds, dim=2)
            
            loss = loss_fn(loss_inp, data['label'], preds_lengths, data['label_len'])
          
            final_preds.append(preds)
            total_loss += loss.item()
            loop.set_postfix({'loss':loss.item()})

    eval_loss = total_loss/num_batches

    return final_preds, eval_loss