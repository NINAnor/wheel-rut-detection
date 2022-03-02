python_venv="your_python_venv"

sudo apt update
sudo apt upgrade

sudo apt install python3-venv python3-dev
sudo apt install gdal-bin
sudo apt install libgdal-dev

python3 -m venv "${python_venv}"

source /home/nibio/python_venv/python_pytorch/bin/activate
pip install -U pip
pip install torch torchvision torchaudio
pip install numpy
pip install matplotlib
pip install pycocotools
pip install wheel
pip install sklearn

gdalinfo --version
# change to your gdal version
pip install gdal==3.0.4

# cuda should be installed. 


