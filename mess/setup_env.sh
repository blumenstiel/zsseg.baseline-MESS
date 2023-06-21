# run script with
# bash mess/setup_env.sh

# Create new environment "zsseg"
conda create --name zsseg -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zsseg

# Install ZSSeg requirements
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip install -r requirements.txt
cd third_party/CLIP && python -m pip install -Ue .

# Install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas