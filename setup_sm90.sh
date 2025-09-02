apt-get update && apt-get install nano zip unzip ffmpeg libsm6 libxext6 unar vim htop gcc curl g++ python3-distutils python3-apt -y

pip install -r requirements.txt

cd generative-models

pip3 install .
pip3 install -r requirements/pt2.txt

pip install 'numpy<2'

mkdir checkpoints
cd checkpoints
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors
cd ../..

