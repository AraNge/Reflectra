from pathlib import Path
from typing import Any, Dict
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reflectra.toml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_nested(config: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)
