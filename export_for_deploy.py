"""Export the champion model + metadata to a lightweight, version-controlled
bundle so the Streamlit dashboard works on Streamlit Cloud (where the MLflow
store mlflow.db / mlruns/ is not available because they are gitignored).

Run locally AFTER training, with the mlops env active:

    python export_for_deploy.py

Produces a `serving_model/` folder containing:
    - model/                 the champion pyfunc model (self-contained)
    - zone_encodings.json    the zone target-encoding maps + priors
    - registry_history.json  the version table the dashboard shows
    - champion.json          {"version": N} for display

Commit `serving_model/` to the repo. It is small (a few MB).
"""

import json
import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "nyc_tip_classification"
OUT_DIR = "serving_model"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    print(f"Champion: {REGISTERED_MODEL_NAME} v{champ.version} (run {champ.run_id})")

    # Fresh output dir
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    # 1) The model itself, copied as a self-contained pyfunc folder.
    model_dst = os.path.join(OUT_DIR, "model")
    local_model = mlflow.artifacts.download_artifacts(
        f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    )
    shutil.copytree(local_model, model_dst)
    print(f"  saved model -> {model_dst}")

    # 2) Zone encodings (downloaded from the champion run).
    enc_path = client.download_artifacts(champ.run_id, "zone_encodings.json")
    shutil.copy(enc_path, os.path.join(OUT_DIR, "zone_encodings.json"))
    print("  saved zone_encodings.json")

    # 3) Registry history table (all versions + their validation metrics).
    history = []
    for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        history.append({
            "version": int(mv.version),
            "val_auc": m.get("val_auc"),
            "val_f1": m.get("val_f1"),
            "val_precision": m.get("val_precision"),
            "val_recall": m.get("val_recall"),
            "is_champion": mv.version == champ.version,
        })
    history.sort(key=lambda r: r["version"])
    with open(os.path.join(OUT_DIR, "registry_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  saved registry_history.json ({len(history)} versions)")

    # 4) Champion pointer.
    with open(os.path.join(OUT_DIR, "champion.json"), "w") as f:
        json.dump({"version": int(champ.version)}, f)
    print("  saved champion.json")

    print(f"\nDone. Commit the '{OUT_DIR}/' folder to the repo.")


if __name__ == "__main__":
    main()
