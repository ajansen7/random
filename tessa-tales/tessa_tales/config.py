"""Load config.toml. TESSA_HOME overrides where the library lives (handy for test runs)."""

import os
import tomllib
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.toml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    home = os.environ.get("TESSA_HOME") or cfg.get("home", "~/TessaTales")
    cfg["home"] = Path(home).expanduser()
    return cfg
