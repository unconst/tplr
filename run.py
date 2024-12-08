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
        parser.add_argument('--netuid', type=int, default=229, help='Bittensor network UID.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--is_validator', action='store_true', help='If validator, turn on to run evals rather than train for incentive.')
        parser.add_argument('--random', action='store_true', help='Trains on a random page instead of correctly assigned.')
        parser.add_argument('--local', action='store_true', help='Gossip through local file system.')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with. e.g., --uids 1 2 3')
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        config = bt.config( parser )
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        return config
    
    def __init__(self):
        # Init config from command line.
        self.config = Miner.config()
        
        # Init bittensor objects.
        self.wallet = bt.wallet( config = self.config )
        self.subtensor = bt.subtensor( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first with: [blue]`btcli subnet register`[/blue]\n')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        tplr.logger.info(f'\n{self.wallet}\n{self.subtensor}\n{self.metagraph}\nuid: {self.uid}')
        
        # Init peers.
        self.peers = list( self.metagraph.uids) if self.config.peers == [] else self.config.peers
        if self.uid not in self.peers: self.peers.append( self.uid ) # Add my self to peers.
        tplr.logger.info(f'peers: {self.peers}')

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
        
        # Init scores.
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        
        # Init wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                for run in wandb.Api().runs(path=self.config.project):
                    if run.name == f'M{self.uid}': run.delete()
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)
        
    # Main training loop.
    async def run( self ):

        # Start background block listener.       
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        # Run until stopped.
        while True:
            
            # Record the window we are on.
            step_window = self.current_window
            # Get the uid to seed data (if validator, take random from peers.)
            step_uid = self.uid if not self.config.is_validator else random.choice([peer for peer in self.config.peers if peer != self.uid])
            tplr.logger.info('\n' + '-' * 40 + f' Window: {step_window} ' + '-' * 40)

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = PAGES_PER_WINDOW,
                seed = self.metagraph.hotkeys[ step_uid ] if not self.config.random else random.randint(0, 10000) # Select seed from step_uid.
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = BATCH_SIZE,
                sequence_length = SEQUENCE_LENGTH,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            tplr.logger.info(f"Pages: {[p[1] for p in pages]} for UID: {step_uid} and Window: {step_window}")
            
            # Accumulate gradient.
            start_time = time.time()
            tplr.logger.info(f"Start accumulating...")
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
                    tplr.logger.info('<Exhuasted window>')
                    break
            tplr.logger.info(f"Stopped accumulating: {i+1} steps, {(i+1) * BATCH_SIZE} bs, and {(i+1) * BATCH_SIZE * SEQUENCE_LENGTH} tokens ")
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
            tplr.logger.info(f"Start gather: {self.peers}")
            response = await tplr.comms.gather(
                state_dict = gradient,
                my_uid = self.uid,
                uids = self.peers,
                window = step_window,
                key = 'gradient',
                timeout = 5,
                device = self.config.device,
                local = self.config.local,
            )
            tplr.logger.info(f"End gather: ({response.time}) - {list(zip(self.peers, response.successes))}")
            if self.config.use_wandb: 
                wandb.log({
                    "total_time": response.time,
                    "upload_bytes": response.upload_bytes,
                    "download_bytes": response.download_bytes,
                    "success_rate": response.success_rate
                })

            # Decompress state and apply to grad.
            for n, p in self.model.named_parameters():
                # Decode grad from all nodes
                if self.config.is_validator:
                    # Get gradient for step uid we are evaluating.
                    eval_idx = response.state_dict[n + 'idxs'][ self.peers.index(step_uid) ]
                    eval_val = response.state_dict[n + 'vals'][ self.peers.index(step_uid) ]
                    # Decompress their gradinet.
                    their_grad = self.transformer.decode(
                        self.compressor.decompress(p, eval_idx, eval_val, xshapes[ n ], totalks[ n ])
                    )
                    # Get my recreated gradient.
                    my_grad = transmitted[ n ]
                    # Compute cosine sim score.
                    score = torch.cdist(their_grad.flatten().unsqueeze(0), my_grad.flatten().unsqueeze(0), p=1).mean()
                    # Compute moving scores and weights.
                    self.scores[step_uid] = SCORES_ALPHA * score + (1 - SCORES_ALPHA) * self.scores[step_uid].expand_as(score)
                    self.weights = torch.softmax(self.scores, dim=0)
                    # Log scores and weights to wandb.
                    if self.config.use_wandb:
                        for uid in self.peers:
                            wandb.log({f"s{uid}": self.scores[uid], f"w{uid}": self.weights[uid] })
                    
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
            tplr.logger.info(f"Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            if self.config.use_wandb: wandb.log({f"lr": self.scheduler.get_last_lr()[0]})

            # Set weights on the chain based on current weights.
            if self.config.is_validator and step_window % WINDOWS_PER_WEIGHTS == 0:
                
                # Set weights on chain.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion = False, # Dont wait, fire and forget.
                    wait_for_finalization = False,
                )
                
            # Check for autoupdate every 360 blocks.
            # if self.current_block % 360 == 0:
            #     tplr.optionally_auto_update( SPEC_VERSION )
            
            # Wait for end of window (if not already done.)
            tplr.logger.info(f"Wait...")
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