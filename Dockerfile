FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

COPY . /app
WORKDIR /app

RUN apt update
RUN pip install --upgrade pip
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

CMD ["tensorboard" , "--logdir=logs" , "--bind_all", "--samples_per_plugin", "images=100"]