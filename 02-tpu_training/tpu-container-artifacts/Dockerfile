FROM python:3.10

WORKDIR /root

# Download and install `tensorflow`.
RUN pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.13.0/tensorflow-2.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Download and install `libtpu`.
# You must save `libtpu.so` in the '/lib' directory of the container image.
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.7.0/libtpu.so -o /lib/libtpu.so

# Download and install tensorflow-datasets
RUN pip3 install tensorflow-datasets tensorboard tensorboard-plugin-profile tensorboard-plugin-wit tensorboard-data-server tensorflow-io
RUN pip3 install google-cloud-aiplatform[cloud_profiler]

# Copies the trainer code to the docker image.
COPY train.py /root/train.py

ENTRYPOINT ["python3", "train.py"]
