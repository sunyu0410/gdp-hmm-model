# Init script
# Based on image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

pip3 install -e /workspace/GDP-HMM_AAPMChallenge/mednext
pip3 install gdown
pip3 install monai
pip3 install opencv-python-headless
pip3 install huggingface_hub