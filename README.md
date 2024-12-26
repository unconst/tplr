
Clone:
```bash
# Clone package
git clone git@github.com:unconst/tplr.git
cd tplr
```

Install requirements:
```bash
# Update 
# Run all common updates
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get autoremove -y
# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# Install npm.
curl -L https://www.npmjs.com/install.sh | sh
# Install pm2.
npm install -g pm2
```

Set up the environment:
```bash
# Create uv venv
uv venv
# Activate venv
source .venv/bin/activate
# Install reqs.
uv sync
```

Run the swarm
```bash
# login to wandb
wandb login
# run the nodes
./start.sh my_run
```