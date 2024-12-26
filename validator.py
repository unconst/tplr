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
VALIDATOR_OFFSET = 2
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
        
        # Init compression.
        self.transformer = tplr.compress.TransformDCT( self.model, target_chunk = TARGET_CHUNK )
        self.compressor = tplr.compress.CompressDCT()
        
        # Init optimizer.
        self.optimizer = optim.SGD(self.model.parameters(), lr = LEARNING_RATE)          
        self.momentum = {}
        self.xshapes = {}
        self.totalks = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
            _, _, xshape, totalk = self.compressor.compress( self.transformer.encode(self.momentum[n]), TOPK_COMPRESSION )
            self.xshapes[n] = xshape; self.totalks[n] = totalk
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10000, T_mult = 1, eta_min = LEARNING_RATE * 0.1)
                
        # Init state params.
        self.step = 0
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int( self.current_block / BLOCKS_PER_WINDOW )
        self.sync_window = self.current_window
                
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
            
            # Wait until we are behind the state by the offset.
            while self.sync_window >= (self.current_window - VALIDATOR_OFFSET):
                print (f'Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{VALIDATOR_OFFSET}')
                time.sleep(12)
                
            # Catch up to the state current - VALIDATOR_OFFSET. 
            # This forces miners to have a retention up to VALIDATOR_OFFSET.
            while self.sync_window < (self.current_window - VALIDATOR_OFFSET):
                
                # Go to next sync.
                self.sync_window += 1
                print (f'Syncing window: {self.sync_window} current: {self.current_window}')
                                
                # Gather gradients from this window.
                step_grads = await tplr.comms.gather(
                    state_dict = {},
                    my_uid = self.config.uid,
                    uids = self.config.peers,
                    window = self.sync_window,
                    key = 'gradient',
                    timeout = 5,
                    device = self.config.device,
                    local = True,
                )

                # Decompress state and apply to gradients
                for n, p in self.model.named_parameters():                
                    new_grad = self.transformer.decode(
                        self.compressor.batch_decompress(
                            p.to(self.config.device), 
                            step_grads.state_dict[n + 'idxs'], 
                            step_grads.state_dict[n + 'vals'], 
                            self.xshapes[n], self.totalks[n]
                        )
                    )
                    # Set recomputed gathered gradient.
                    if p.grad is None: p.grad = new_grad
                    else: p.grad.copy_(new_grad)
                    p.grad.sign_()
                        
                # Apply the optimizer step
                self.optimizer.step()
                self.scheduler.step()
                if self.config.use_wandb: wandb.log({f"lr": self.scheduler.get_last_lr()[0]})
                
            # Get a random peer to eval on their gradient at self.sync_window + 1
            eval_uid = random.choice( self.config.peers )
            # Get the pages for the window infront of the current sync window
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = self.sync_window + 1,
                n_pages = PAGES_PER_WINDOW,
                seed = eval_uid
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = BATCH_SIZE,
                sequence_length = SEQUENCE_LENGTH,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            print (f'Evalling uid: {eval_uid} on window: {self.sync_window + 1} with state from: {self.sync_window} and pages: {[p[1] for p in pages]}')
            
            # Get loss on all samples from this window.
            loss_before = 0
            for i, batch in enumerate( loader ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                loss_before += self.model(input_ids=input_ids, labels=labels).loss.item()
            print (f'Computed total loss before: {loss_before}')
                                
            # Get the gradients from this miner on this window.
            eval_grad = await tplr.comms.get(
                uid = eval_uid, 
                window = self.sync_window + 1, 
                key = 'gradient', 
                timeout = 5, 
                local = True, 
            )
            if eval_grad == None:
                score = 0
                print (f'Miner with uid: {eval_uid} has no gradient for window: {self.sync_window + 1}')
                continue

            # Apply grad to model which is at state sync_window
            for n, p in self.model.named_parameters():  
                # Decompress their gradient.
                decompressed_grad = self.transformer.decode( 
                    self.compressor.decompress(
                        p.to(self.config.device),
                        eval_grad[n + 'idxs'].to(self.config.device), 
                        eval_grad[n + 'vals'].to(self.config.device),
                        self.xshapes[n], self.totalks[n],
                    )
                )
                # Apply this grad to the param of the model using the learning rate of the scheduler
                p.data.sub_(decompressed_grad, alpha = self.scheduler.get_last_lr()[0] ) 
                
            # Get loss after we apply the gradient.
            loss_after = 0
            for i, batch in enumerate( loader ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                loss_after += self.model(input_ids=input_ids, labels=labels).loss.item()
            print (f'Computed total loss after: {loss_after}')
     
            # Remove gradient from the model
            for n, p in self.model.named_parameters():  
                # Decompress their gradient.
                decompressed_grad = self.transformer.decode( 
                    self.compressor.decompress(
                        p.to(self.config.device),
                        eval_grad[n + 'idxs'].to(self.config.device), 
                        eval_grad[n + 'vals'].to(self.config.device),
                        self.xshapes[n], self.totalks[n],
                    )
                )
                # Apply this grad to the param of the model using the learning rate of the scheduler
                p.data.add_(decompressed_grad, alpha = self.scheduler.get_last_lr()[0] ) 
                
            # Compute score
            score = loss_before - loss_after
            print (f'score: {score}')
    
            
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