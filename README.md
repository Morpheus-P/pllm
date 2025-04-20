# pllm
Prepare for LLM

llama.cpp Installation
2025/3/27



## Install and compile toolchain:

sudo apt update && sudo apt install -y git cmake make python3 python3-pip g++ wget  

## Installation and Configure CUDA:

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

## Add CUDA path to ~/.bashrc :

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc  
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc  
source ~/.bashrc‌ 

## Check cuda-toolkit g++ cmake Installation:

nvcc -V
##nvcc: NVIDIA (R) Cuda compiler driver
##Copyright (c) 2005-2025 NVIDIA Corporation
##Built on Fri_Feb_21_20:23:50_PST_2025
##Cuda compilation tools, release 12.8, V12.8.93
##Build cuda_12.8.r12.8/compiler.35583870_0

g++ --version
##g++ (Debian 12.2.0-14) 12.2.0
##Copyright (C) 2022 Free Software Foundation, Inc.
##This is free software; see the source for copying conditions.  There is NO
##warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

cmake --version
##cmake version 3.25.1
##
##CMake suite maintained and supported by Kitware (kitware.com/cmake).

## Creating Virtual Environment (Python 3)

python3 -m venv llpp
source llpp/bin/activate

## Clone llama.cpp:

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp && mkdir build && cd build 

cmake .. -DGGLM_CUDA=ON -DGGLM_CCACHE=OFF
cmake --build . --config Release

# Download LLM Model:

mkdir -p models && cd models  
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

https://hf-mirror.com/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/blob/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf

Instructions to run this model in llama.cpp:
You can view more detailed instructions in our blog: unsloth.ai/blog/deepseek-r1

Do not forget about <｜User｜> and <｜Assistant｜> tokens! - Or use a chat template formatter

Obtain the latest llama.cpp at https://github.com/ggerganov/llama.cpp

Example with Q8_0 K quantized cache Notice -no-cnv disables auto conversation mode

./llama.cpp/llama-cli \
    --model unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
    --cache-type-k q8_0 \
    --threads 16 \
    --prompt '<｜User｜>What is 1+1?<｜Assistant｜>' \
    -no-cnv


./llama.cpp/llama-cli \
--model unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
--cache-type-k q8_0 
--threads 16 
--prompt '<｜User｜>What is 1+1?<｜Assistant｜>'
--n-gpu-layers 20 \
 -no-cnv

