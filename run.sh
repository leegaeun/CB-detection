#!/bin/bash

docker build -t cbdetection:2.2.0 -f Dockerfile .

nvidia-docker run -it --gpus all --rm               \
	-v "$PWD"/data:/data                            \
	-v "$PWD"/src:/src                              \
	--name cont.cbdetection                         \
	-p 8892-8893:8892-8893                          \
	cbdetection:2.2.0                               \
	sh -c 'jupyter notebook --allow-root --port 8892'
