import os
from pathlib import Path
from ultralytics import YOLO


def download_yolo():
    print("=" * 60)
    print("        SURDAS AI - YOLOv8 MODEL SETUP")
    print("=" * 60)

    # Create a writable models directory
    project_dir = Path(__file__).resolve().parent
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "yolov8n.pt"

    print("\n[1/1] Downloading YOLOv8 Nano (yolov8n.pt)...")
    print(f"      Save location: {model_path}")

    try:
        # If model already exists, load it
        if model_path.exists():
            print("  -> Model already exists.")
            model = YOLO(str(model_path))

        else:
            # Download YOLOv8n
            model = YOLO("yolov8n.pt")

            # Find the downloaded model and move it to our models folder
            downloaded_model = Path("yolov8n.pt")

            if downloaded_model.exists():
                downloaded_model.replace(model_path)

            # Reload from the final location
            model = YOLO(str(model_path))

        print("\n  -> SUCCESS: YOLOv8n is ready!")
        print(f"  -> Model: {model_path}")

    except Exception as e:
        print("\n  -> ERROR in YOLOv8:")
        print(f"     {e}")

    print("\n" + "=" * 60)
    print("        YOLOv8 SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    download_yolo()