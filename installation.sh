pip3 install torch torchvision torchaudio
pip install pytorch-lightning==1.2.8



export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA=cu102
export TORCH=1.11.0

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install pytorch-metric-learning


conda install -c plotly psutil requests python-kaleido --yes
pip install cython==0.29.20 autowrap ninja tables ply ilock
pip install h5py pydocstyle plotly psutil xvfbwrapper yapf mypy openmesh plyfile neuralnet-pytorch imageio pyinstrument pairing robust_laplacian pymesh trimesh cmake "ray[tune]" "pytorch-lightning-bolts>=0.2.5" pyrr gdist neptune-client neptune-contrib iopath sklearn autowrap py-goicp opencv-python torchsummary gdown
conda install "notebook>=5.3" "ipywidgets>=7.2" flake8 black flake8 -y
pip install addict
pip install open3d-python
pip install git+https://github.com/fwilliams/point-cloud-utils
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
git rm -rf ChamferDistancePytorch
rm -rf ChamferDistancePytorch
git submodule add --force https://github.com/ThibaultGROUEIX/ChamferDistancePytorch