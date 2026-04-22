import os

import docker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_names = ["bert", "roberta", "distilbert", "deberta", "electra"]


def main():
    print("[not-docker] creating docker client...")
    client = docker.from_env()
    print("[not-docker] docker client created.")
    print("[docker] building docker image...")
    for chunk in client.api.build(
        path=project_root,
        dockerfile="encoders/Dockerfile",
        tag="sarcasm-encoder",
        decode=True,
    ):
        if "stream" in chunk:
            print(chunk["stream"], end="")
        if "error" in chunk:
            print("ERROR:", chunk["error"])
            return

    print("[docker] docker image built.")

    for model in model_names:
        print(f"[{model}] starting...")
        try:
            container_output = client.containers.run(
                "sarcasm-encoder",
                environment={
                    "ENCODER": model,
                    "TRAIN_DATA": "/app/data/news_headlines_train.tsv",
                    "VAL_DATA": "/app/data/news_headlines_val.tsv",
                    "PYTHONUNBUFFERED": "1",
                },
                volumes={
                    os.path.join(project_root, "data", "processed"): {
                        "bind": "/app/data",
                        "mode": "ro",
                    },
                    os.path.join(project_root, "encoders", "outputs"): {
                        "bind": "/app/outputs",
                        "mode": "rw",
                    },
                },
                remove=True,  # is essentially the --rm flag
                stream=True,
                stderr=True,
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
            )
            for chunk in container_output:
                print(chunk.decode(), end="")
            print(f"[{model}] done.")
        except docker.errors.ContainerError as e:
            print(f"[{model}] FAILED (exit {e.exit_status})")
            print(e.stderr.decode() if e.stderr else "no stderr")
            break

    print("Done.")


if __name__ == "__main__":
    main()
