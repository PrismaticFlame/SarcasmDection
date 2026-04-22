# Build context must be the encoders/ directory:
#   docker build -f encoders/Dockerfile encoders/
# or from within encoders/:
#   docker build .
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY encoders/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY encoders/ .

COPY common/ ./common/

VOLUME ["/app/outputs"]

ENV ENCODER=bert

ENTRYPOINT ["sh", "-c", "python train.py --encoder ${ENCODER}"]

