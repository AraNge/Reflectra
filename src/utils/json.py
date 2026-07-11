from __future__ import annotations

import json as json_lib
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


JsonObject = dict[str, Any]


def read_json(path: str | Path, missing_ok: bool = False) -> JsonObject:
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        value = json_lib.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return value


def write_json(path: str | Path, value: JsonObject) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json_lib.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")

    temporary_path.replace(path)


def stream_jsonl(
    path: str | Path,
    missing_ok: bool = False,
) -> Iterator[JsonObject]:
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json_lib.loads(line)
            except json_lib.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )

            yield value


def read_jsonl(
    path: str | Path,
    missing_ok: bool = False,
) -> list[JsonObject]:
    return list(stream_jsonl(path, missing_ok=missing_ok))


def append_jsonl(path: str | Path, records: Iterable[JsonObject]) -> None:
    records = list(records)
    if not records:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(
                json_lib.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

        file.flush()
        os.fsync(file.fileno())


def write_jsonl(
    path: str | Path,
    records: Iterable[JsonObject],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json_lib.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    temporary_path.replace(path)
