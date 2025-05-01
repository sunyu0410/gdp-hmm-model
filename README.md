# Docker

## Build
* Isolate `Dockerfile` and `verify.ipynb`. Move everthing else to `src`. 
* `docker build . -t gdp-hmm:v1`

## Run
`docker run -it --gpus=all -v <data dir>:/data -v <result dir>:/results gdp-hmm:v1`

It will run `python inference.py <data dir> <result dir>`.

