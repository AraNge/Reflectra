import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dictionaries.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {e}"
                ) from e

    return rows


def stream_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """
    Stream a JSONL file row by row.
    Useful for very large metadata files.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {e}"
                ) from e


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write list of dictionaries to JSONL.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")