import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import os
import numpy as np
from Llama2 import *
from CustomDataset import TextDataset

torch.manual_seed(0)
# Config
LEARNIG_RATE = 5e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
MAX_SEQ_LEN = 1024
CHECKPOINT_DIR = "./training_output"
CHECKPOINT_NAME = "best_ckpt.pth"
allow_cuda = False
device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

# Load model and Optimizer
print("Loading Llama2 model & Adam optimizer...")
model = LLaMA.build(
    checkpoints_dir='llama-2-7b',
    tokenizer_path='tokenizer.model',
    load_model=False,
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=BATCH_SIZE,
    device=device
)

optim = torch.optim.Adam(model.model.parameters(), lr=LEARNIG_RATE)
best_loss = 100.0

# Load loss function
print("Loading loss function...")
criterion = nn.NLLLoss()

# Data loader
print("Loading data...")
csv_path = 'data\\tinystories1k.csv'
data = TextDataset(csv_path, model.tokenizer)
train_dataset, val_dataset = random_split(data, [int(0.9 * len(data)), len(data) - int(0.9 * len(data))])
train_dataloader = DataLoader(train_dataset, batch_size=model.args.max_batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=model.args.max_batch_size)

# Training
def valid(model, data):
    model.eval()

    valid_losses = []

    with torch.no_grad():
        bar = tqdm(enumerate(data), total=len(data), desc="VALIDATION")
        for _, batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)

            logits = model.model(X)
            logits = F.log_softmax(logits, dim=-1)

            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                y.reshape(-1)
            )

            valid_losses.append(loss.item())
            bar.set_postfix(TRAIN="Batch_Loss {:.2f} - Valid_Loss {:.2f}".format(
                    loss.item(),
                    np.mean(valid_losses)
                    )
                )
            
            del X, y, logits
            torch.cuda.empty_cache()

    # mean_valid_loss = np.mean(valid_losses)
    return np.mean(valid_losses)

print("Start training ...")

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"#################### Epoch: {epoch} ####################")

    model.model.train()
    train_losses = []
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TRAINING")

    for _, batch in bar:
        X, y = batch
        X, y = X.to(device), y.to(device)

        logits = model.model(X)
        logits = F.log_softmax(logits, dim=-1)

        optim.zero_grad()
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            y.reshape(-1)
        )

        loss.backward()
        optim.step()

        train_losses.append(loss.item())

        del X, y, logits
        torch.cuda.empty_cache()

        bar.set_postfix(TRAIN="Epoch {} - Batch_Loss {:.2f} - Train_Loss {:.2f} - Best_Valid_Loss {:.2f}".format(
                    epoch,
                    loss.item(),
                    np.mean(train_losses),
                    best_loss
                    )
        )

    valid_loss = valid(model.model, val_dataloader)

    if valid_loss < best_loss:
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        best_loss = valid_loss
        # state_dict = {
        #     'model_state_dict': model.model.state_dict(),
        #     'optim_state_dict': optim.state_dict(),
        #     'loss': best_loss
        # }
        torch.save(model.model.state_dict(), f"{CHECKPOINT_DIR/{CHECKPOINT_NAME}}")
        print(f"***** Current best checkpoint is saved. *****")

    print(f"Valid loss: {valid_loss} - Best valid loss {best_loss}")
    
print(f"Training finished!")