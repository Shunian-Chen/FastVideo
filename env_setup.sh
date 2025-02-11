#!/bin/bash

# install torch
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements-lint.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# install fastvideo
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
