# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Global imports.
import sys
import time
import wandb
import torch
import random
import asyncio
import argparse
import threading
import numpy as np
import bittensor as bt
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, LlamaConfig
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import local package.
import tplr

# GPU optimizations.
# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Globals: determined by master.
SPEC_VERSION = 5 # Run version.
PROJECT = f'llama-demo' # wandb project.
SEQUENCE_LENGTH = 1024 # global sequence length.
PAGES_PER_WINDOW = 5 # Pages to train on (and be evaluated on each window.)
BATCH_SIZE = 8 # global batch size.
WEIGHT_DECAY = 0.1
LEARNING_RATE = 4e-4 # global learning rate.
BLOCKS_PER_WINDOW = 2 # blocks per window step.
WINDOWS_PER_SYNC = 100 # Step Windows before sync state occurs.
MOMENTUM_DECAY = 0.999 # momentum deacy rate.
TOPK_COMPRESSION = 32 # DeMo Topk Compression.
TARGET_CHUNK = 64 # DeMo chunk size.
SCORES_ALPHA = 0.0001 # Scores moving average.
WINDOWS_PER_WEIGHTS = 10 # Windows before validator sets weights on chain.

tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/LLaMA-2-7B-32K", verbose=False, clean_up_tokenization_spaces=True
)
tokenizer.pad_token = tokenizer.eos_token

model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=8,
    intermediate_size=8192,
    num_key_value_heads=8,
    activation_function="swiGLU",
    max_position_embeddings=2048,
)

class Miner:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default=PROJECT, help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with. e.g., --uids 1 2 3')
        parser.add_argument('--uid', type=int, default=229, help='This Peer uid.')
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        config = bt.config( parser )
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        return config
    
    def __init__(self):
        # Init config from command line.
        self.config = Miner.config()      
        self.subtensor = bt.subtensor(config = self.config)  
        
        # Initialize the model with the same seed so that all workers have the same model at init.
        self.model = LlamaForCausalLM(config=model_config)
        self.model.to(self.config.device)
        self.model.train()
        
        # Init tokenizer.
        self.tokenizer = tokenizer
        
        # Init optimizer.
        self.optimizer = optim.SGD(self.model.parameters(), lr = LEARNING_RATE)          
        self.momentum = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10000, T_mult = 1, eta_min = LEARNING_RATE * 0.1)
        
        # Init compression.
        self.transformer = tplr.compress.TransformDCT( self.model, target_chunk = TARGET_CHUNK )
        self.compressor = tplr.compress.CompressDCT()
        
        # Init state params.
        self.step = 0
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int( self.current_block / BLOCKS_PER_WINDOW )
                
        # Init wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                for run in wandb.Api().runs(path=self.config.project):
                    if run.name == f'M{self.config.uid}': run.delete()
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.config.uid}', config=self.config)
        
    # Main training loop.
    async def run( self ):

        # Start background block listener.       
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        # Run until stopped.
        while True:
            
            # Record the window we are on.
            step_window = self.current_window
            print('\n' + '-' * 40 + f' Window: {step_window} ' + '-' * 40)

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = PAGES_PER_WINDOW,
                seed = self.config.uid
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = BATCH_SIZE,
                sequence_length = SEQUENCE_LENGTH,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            print(f"Pages: {[p[1] for p in pages]} for UID: {self.config.uid} and Window: {step_window}")
            
            # Accumulate gradient.
            start_time = time.time()
            print(f"Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            for i, batch in enumerate( loader ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                outputs = self.model(input_ids=input_ids, labels=labels)
                outputs.loss.backward()
                print ('loss:', outputs.loss.item())
                if self.current_window != step_window:
                    print('<Exhuasted window>')
                    break
            print(f"Stopped accumulating: {i+1} steps, {(i+1) * BATCH_SIZE} bs, and {(i+1) * BATCH_SIZE * SEQUENCE_LENGTH} tokens ")
            duration = time.time() - start_time
            if self.config.use_wandb: wandb.log({f"loss": outputs.loss.item(), "bs": (i+1) * BATCH_SIZE, "toks": ((i+1) * BATCH_SIZE * SEQUENCE_LENGTH)/duration })
                
            # Reduce gradient using DeMo.
            gradient = {}
            xshapes = {}
            totalks = {}
            transmitted = {}
            for n, p in self.model.named_parameters():
                # Step-Weight decay
                p.data.mul_( 1.0 - self.scheduler.get_last_lr()[0] * WEIGHT_DECAY )
                # Momentum decay
                self.momentum[n].mul_( MOMENTUM_DECAY )
                # Add the grad to the momentum.
                self.momentum[n].add_( p.grad, alpha=self.scheduler.get_last_lr()[0] )
                # Compress gradient.
                idxs, vals, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(self.momentum[n]), TOPK_COMPRESSION
                )
                # Estimate transmitted gradient.
                transmit_grad = self.transformer.decode(
                    self.compressor.decompress(p, idxs, vals, xshape, totalk)
                )
                # Remove the transmitted from delta (double couting)
                self.momentum[n].sub_(transmit_grad)
                # Add to share_state
                transmitted[ n ] = transmit_grad
                gradient[ n + 'idxs'] = idxs 
                gradient[ n + 'vals'] = vals
                xshapes[ n ] = xshape; totalks[ n ] = totalk

            # All-gather share state from all peers with timeout.
            print(f"Start gather: {self.config.peers}")
            response = await tplr.comms.gather(
                state_dict = gradient,
                my_uid = self.config.uid,
                uids = self.config.peers,
                window = step_window,
                key = 'gradient',
                timeout = 5,
                device = self.config.device,
                local = True,
            )
            print(f"End gather: ({response.time}) - {list(zip(self.config.peers, response.successes))}")
            if self.config.use_wandb: 
                wandb.log({
                    "total_time": response.time,
                    "upload_bytes": response.upload_bytes,
                    "download_bytes": response.download_bytes,
                    "success_rate": response.success_rate
                })

            # Decompress state and apply to grad.
            for n, p in self.model.named_parameters():                
                # Decompress all gradients in batch form to produce shared gradient.
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p, response.state_dict[n + 'idxs'], response.state_dict[n + 'vals'], xshapes[ n ], totalks[ n ]
                    )
                )
                # new_grad += self.momentum[n]
                # Set recomputed gathered gradient.
                if p.grad is None: p.grad = new_grad
                else: p.grad.copy_(new_grad)
                # Sign-SGD
                p.grad.sign_()
                    
            # Apply the optimizer step
            print(f"Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            if self.config.use_wandb: wandb.log({f"lr": self.scheduler.get_last_lr()[0]})

            # Wait for end of window (if not already done.)
            print(f"Wait...")
            while self.current_window == step_window: time.sleep(0.1)
            self.step += 1
            
    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            if int( self.current_block / BLOCKS_PER_WINDOW ) != self.current_window:
                self.current_window = int( self.current_block / BLOCKS_PER_WINDOW ) 
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                time.sleep(1) 

# Start miner/validator.
if __name__ == "__main__":
    asyncio.run( Miner().run() )