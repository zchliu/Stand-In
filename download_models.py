from huggingface_hub import snapshot_download


def main():
    snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="checkpoints/base_model/")
    snapshot_download(
        "DIAMONIK7777/antelopev2", local_dir="checkpoints/antelopev2/models/antelopev2"
    )
    snapshot_download("BowenXue/Stand-In", local_dir="checkpoints/Stand-In/")


if __name__ == "__main__":
    main()
