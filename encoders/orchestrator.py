import argparse
import os

import docker
import docker.errors

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_names = ["bert", "roberta", "distilbert", "deberta", "electra"]
datasets = ["csc", "isarcasmeval", "mustard", "news_headlines", "sarc", "sarcasm_v2"]

IMAGE_TAG = "sarcasm-encoder"


def build_image(client):
    print("[docker] building docker image...")
    for chunk in client.api.build(
        path=project_root,
        dockerfile="encoders/Dockerfile",
        tag=IMAGE_TAG,
        decode=True,
    ):
        if "stream" in chunk:
            print(chunk["stream"], end="")
        if "error" in chunk:
            print("ERROR:", chunk["error"])
            return False
    print("[docker] docker image built.")
    return True


def image_exists(client):
    try:
        client.images.get(IMAGE_TAG)
        return True
    except docker.errors.ImageNotFound:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the Docker image")
    args = parser.parse_args()

    print("[not-docker] creating docker client...")
    client = docker.from_env()
    print("[not-docker] docker client created.")

    if args.rebuild or not image_exists(client):
        if not build_image(client):
            return
    else:
        print(f"[docker] image '{IMAGE_TAG}' already exists, skipping build (use --rebuild to force)")

    volumes = {
        os.path.join(project_root, "data", "processed"): {
            "bind": "/app/data",
            "mode": "ro",
        },
        os.path.join(project_root, "encoders", "outputs"): {
            "bind": "/app/outputs",
            "mode": "rw",
        },
    }
    gpu = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    for model in model_names:
        print(f"[{model}] training (all datasets)...")
        try:
            container_output = client.containers.run(
                IMAGE_TAG,
                environment={
                    "ARGS": f"--encoder {model}",
                    "PYTHONUNBUFFERED": "1",
                },
                volumes=volumes,
                remove=True,
                stream=True,
                stderr=True,
                device_requests=gpu,
            )
            for chunk in container_output:
                print(chunk.decode(), end="")
            print(f"[{model}] training done.")
        except docker.errors.ContainerError as e:
            print(f"[{model}] FAILED (exit {e.exit_status})")
            print(e.stderr if e.stderr else "no stderr")

    print("All training complete. Starting cross-dataset evaluation...")

    for model in model_names:
        print(f"[{model}] cross-dataset evaluation...")
        try:
            container_output = client.containers.run(
                IMAGE_TAG,
                environment={
                    "SCRIPT": "common/cross_dataset.py",
                    "ARGS": f"--encoder {model}",
                    "PYTHONUNBUFFERED": "1",
                },
                volumes=volumes,
                remove=True,
                stream=True,
                stderr=True,
                device_requests=gpu,
            )
            for chunk in container_output:
                print(chunk.decode(), end="")
            print(f"[{model}] cross-dataset done.")
        except docker.errors.ContainerError as e:
            print(f"[{model}] cross FAILED (exit {e.exit_status})")
            print(e.stderr if e.stderr else "no stderr")

    print("Done.")


if __name__ == "__main__":
    main()
