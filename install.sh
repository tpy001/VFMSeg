conda create -n rein -y
conda activate rein
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install cityscapesscripts==2.2.0
pip install -U openmim
# python -m pip install pip~=19.0  # If there are issues installing mmengine, please downgrade your pip version by uncommenting this line.
mim install mmengine==0.10.4
mim install "mmcv==2.1.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt


# Another way

# conda env create -f environment.yml

# pip install -U openmim
# mim install mmengine
# mim install "mmcv==2.1.0"
# pip install "mmsegmentation>=1.0.0"
# pip install "mmdet>=3.0.0"
