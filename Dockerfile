# Dockerfile for GDP-HMM Challenge
# Yu Sun
# yu.sun@petermac.org
#

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

COPY ./src ./
COPY ./wheel /wheel
COPY ./Dockerfile ./src/requirements.txt /
RUN pip install /wheel/*.whl
RUN rm -rf /wheel

RUN pip install monai
RUN pip install -e ./mednext

CMD python inference.py /data /results

