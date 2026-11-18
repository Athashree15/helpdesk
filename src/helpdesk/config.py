"""Configuration loading. All paths are resolved relative to the project root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides (handy for quick experiments without editing the file)
    o = cfg["ollama"]
    o["base_url"] = os.getenv("OLLAMA_BASE_URL", o["base_url"])
    o["chat_model"] = os.getenv("HELPDESK_CHAT_MODEL", o["chat_model"])
    o["embed_model"] = os.getenv("HELPDESK_EMBED_MODEL", o["embed_model"])

    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((PROJECT_ROOT / value).resolve())

    uri = cfg["llmops"]["mlflow_tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///"):]
        abs_path = (PROJECT_ROOT / rel).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        cfg["llmops"]["mlflow_tracking_uri"] = "sqlite:///" + abs_path.as_posix()

    Path(cfg["paths"]["storage_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def load_prompts(cfg: dict[str, Any]) -> dict[str, Any]:
    with open(cfg["paths"]["prompts_file"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
