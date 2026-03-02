FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --upgrade pip && \
    pip install runpod diffusers==0.30.2 transformers==4.44.2 accelerate==0.34.2 sentencepiece huggingface_hub protobuf pillow peft
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
