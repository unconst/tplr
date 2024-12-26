
pm2 delete all

pm2 start miner.py --interpreter python3 --name M0 -- --device cuda:0 --use_wandb --uid 0 --peers 0 1 2 3 4 --project $1
pm2 start miner.py --interpreter python3 --name M1 -- --device cuda:1 --use_wandb --uid 1 --peers 0 1 2 3 4 --project $1
pm2 start miner.py --interpreter python3 --name M2 -- --device cuda:2 --use_wandb --uid 2 --peers 0 1 2 3 4 --project $1
pm2 start miner.py --interpreter python3 --name M3 -- --device cuda:3 --use_wandb --uid 3 --peers 0 1 2 3 4 --project $1
pm2 start miner.py --interpreter python3 --name M4 -- --device cuda:4 --use_wandb --uid 4 --peers 0 1 2 3 4 --project $1
pm2 start validator.py --interpreter python3 --name V5 -- --device cuda:5 --use_wandb --uid 5 --peers 0 1 2 3 4 --project $1
