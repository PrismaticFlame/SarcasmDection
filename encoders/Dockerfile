# Build context must be the encoders/ directory:
#   docker build -f encoders/Dockerfile encoders/
# or from within encoders/:
#   docker build .
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/outputs"]

ENV ENCODER=bert

ENTRYPOINT ["sh", "-c", "python train.py --encoder ${ENCODER}"]

