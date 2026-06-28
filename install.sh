python -m venv env_isaaclab
# activate the virtual environment
source env_isaaclab/bin/activate

pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128