# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import time
import torch
import random
import asyncio
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Local imports
import tplr

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Default hyperparameters (you can modify via command line)
DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_PAGES_PER_WINDOW = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 1
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

async def load_data(tokenizer, sequence_length, pages_per_window, batch_size, seed=42, offset=0):
    # Retrieve pages using tplr's dataset loader, similar to the distributed script
    pages = await tplr.dataset.DatasetLoader.next_pages(
        offset=offset,
        n_pages=pages_per_window,
        seed=seed
    )
    loader = await tplr.dataset.DatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages,
        tokenizer=tokenizer
    )
    return pages, loader

async def main():
    parser = argparse.ArgumentParser(description='Single-node baseline GPT-2 training')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device: cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--pages_per_window', type=int, default=DEFAULT_PAGES_PER_WINDOW, help='Pages per window (dataset chunk)')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQUENCE_LENGTH, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data selection')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2LMHeadModel.config_class())
    model.to(args.device)
    
    # Use AdamW for optimization
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=1)

    # Just an example of multiple "windows" or steps
    # For the baseline, we can simply loop over a number of offsets or just do multiple epochs on the same offset
    global_step = 0
    for epoch in range(args.epochs):
        # Load one "window" of data
        pages, loader = await load_data(
            tokenizer=tokenizer,
            sequence_length=args.seq_len,
            pages_per_window=args.pages_per_window,
            batch_size=args.batch_size,
            seed=args.seed,
            offset=epoch  # you can vary this to get different pages each epoch
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}: Training on pages {[p[1] for p in pages]}")
        model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            input_ids = torch.tensor(batch, dtype=torch.long).to(args.device)
            labels = input_ids.clone()
            labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping trick
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if (i+1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                elapsed = time.time() - start_time
                print(f"  Step {i+1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Speed: {(num_batches*args.batch_size*args.seq_len)/elapsed:.2f} tokens/s")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nFinished epoch {epoch+1}. Average Loss: {avg_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    asyncio.run(main())
