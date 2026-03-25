# Atari
pip install gym==0.19.0
pip install atari-py==0.2.9
pip install opencv-python==4.7.0.72

# ROMs
mkdir roms && cd roms
wget -L -nv http://www.atarimania.com/roms/Roms.rar
unrar x -o+ Roms.rar
python -m atari_py.import_roms ROMS
cd .. && rm -rf roms

# DMC
pip install dm_control

# PyTorch
pip install torch==2.0.1
pip install torchvision==0.15.2

# Numpy
pip install numpy==1.23.5

# Other
pip install tqdm
pip install wandb
pip install av
pip install tensorboard