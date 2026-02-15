#!/usr/bin/env python3
"""
Build a local metadata database for League splash artworks.

Key behavior:
- Pulls `Module:SkinData/data` and module revision metadata from the LoL wiki.
- Parses all champion/skin metadata from Lua table syntax.
- Resolves candidate splash image URLs (HD first, then regular) without downloading.
- Groups entries by resolved URL to detect shared splash artworks.
 - Writes:
  - metadata.json (full database + source/version/diff information)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


TOOL_VERSION = "0.1.1"
DEFAULT_BASE_URL = "https://wiki.leagueoflegends.com/en-us"
SKINDATA_MODULE = "Module:SkinData/data"
FILENAME_MODULE = "Module:Filename"
REQUEST_TIMEOUT = 60
RETRY_ATTEMPTS = 4


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    # Windows-safe path segment.
    value = re.sub(r'[<>:"/\\\\|?*]+', "_", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value or "_"


def stable_hash(value: str, size: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:size]


def backup_metadata_file(metadata_path: Path, output_dir: Path) -> Optional[Path]:
    if not metadata_path.exists():
        return None
    backup_root = output_dir / "backups" / "metadata"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_name = f"metadata.{stamp}.json"
    backup_path = backup_root / backup_name
    shutil.copy2(metadata_path, backup_path)
    return backup_path


def request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    delay = 1.0
    last_error: Optional[Exception] = None
    for _ in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed request after retries: {url} params={params} error={last_error}")


def request_text(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    delay = 1.0
    last_error: Optional[Exception] = None
    for _ in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed request after retries: {url} params={params} error={last_error}")


def encode_filename_piece(text: str) -> str:
    # Mirrors Module:Filename encode()
    return text.replace(":", "-").replace("/", "")


def title_key(file_title: str) -> str:
    # Normalize for dictionary lookups.
    return file_title.replace(" ", "_")


def build_base_title(champion: str, skin: str) -> str:
    champion_encoded = encode_filename_piece(champion)
    skin_encoded = encode_filename_piece(skin).replace(" ", "")
    return f"{champion_encoded}_{skin_encoded}Skin"


class LuaParseError(RuntimeError):
    pass


class LuaTableParser:
    """
    Small Lua table parser for the SkinData module.
    Supports strings, numbers, booleans, nil, tables, keyed fields, and arrays.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.pos = 0

    def parse(self) -> Any:
        self._skip_ws_comments()
        if self._peek_word("return"):
            self.pos += 6
        self._skip_ws_comments()
        value = self._parse_value()
        self._skip_ws_comments()
        return value

    def _peek(self) -> str:
        if self.pos >= self.length:
            return ""
        return self.source[self.pos]

    def _peek_word(self, word: str) -> bool:
        return self.source[self.pos : self.pos + len(word)] == word

    def _advance(self, count: int = 1) -> None:
        self.pos += count

    def _skip_ws_comments(self) -> None:
        while self.pos < self.length:
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue
            if self.source.startswith("--", self.pos):
                while self.pos < self.length and self.source[self.pos] != "\n":
                    self._advance()
                continue
            break

    def _expect(self, expected: str) -> None:
        self._skip_ws_comments()
        if not self.source.startswith(expected, self.pos):
            got = self.source[self.pos : self.pos + 20]
            raise LuaParseError(f"Expected {expected!r} at {self.pos}, got {got!r}")
        self._advance(len(expected))

    def _parse_string(self) -> str:
        quote = self._peek()
        if quote not in ("'", '"'):
            raise LuaParseError(f"Expected quote at {self.pos}")
        self._advance()
        out: List[str] = []
        while self.pos < self.length:
            ch = self._peek()
            self._advance()
            if ch == quote:
                return "".join(out)
            if ch == "\\":
                if self.pos >= self.length:
                    break
                nxt = self._peek()
                self._advance()
                escapes = {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "\\": "\\",
                    "'": "'",
                    '"': '"',
                }
                out.append(escapes.get(nxt, nxt))
            else:
                out.append(ch)
        raise LuaParseError("Unterminated string literal")

    def _parse_number(self) -> Any:
        start = self.pos
        while self.pos < self.length and self._peek() in "+-0123456789.eE":
            self._advance()
        text = self.source[start : self.pos]
        if re.match(r"^[+-]?\d+$", text):
            return int(text)
        try:
            return float(text)
        except ValueError as exc:
            raise LuaParseError(f"Invalid number {text!r} at {start}") from exc

    def _parse_identifier(self) -> str:
        start = self.pos
        while self.pos < self.length and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        if self.pos == start:
            raise LuaParseError(f"Expected identifier at {self.pos}")
        return self.source[start : self.pos]

    def _parse_value(self) -> Any:
        self._skip_ws_comments()
        ch = self._peek()
        if not ch:
            raise LuaParseError("Unexpected end while parsing value")
        if ch == "{":
            return self._parse_table()
        if ch in ("'", '"'):
            return self._parse_string()
        if ch.isdigit() or ch in "+-":
            return self._parse_number()
        ident = self._parse_identifier()
        if ident == "true":
            return True
        if ident == "false":
            return False
        if ident == "nil":
            return None
        return ident

    def _parse_table(self) -> Any:
        self._expect("{")
        keyed: Dict[Any, Any] = {}
        array: List[Any] = []

        while True:
            self._skip_ws_comments()
            if self._peek() == "}":
                self._advance()
                break

            key: Optional[Any] = None
            value: Any

            if self._peek() == "[":
                self._advance()
                self._skip_ws_comments()
                if self._peek() in ("'", '"'):
                    key = self._parse_string()
                else:
                    key = self._parse_value()
                self._skip_ws_comments()
                self._expect("]")
                self._skip_ws_comments()
                self._expect("=")
                value = self._parse_value()
                keyed[key] = value
            else:
                checkpoint = self.pos
                try:
                    ident = self._parse_identifier()
                    self._skip_ws_comments()
                    if self._peek() == "=":
                        self._advance()
                        value = self._parse_value()
                        keyed[ident] = value
                    else:
                        # It was a value token, not key=value.
                        self.pos = checkpoint
                        value = self._parse_value()
                        array.append(value)
                except LuaParseError:
                    self.pos = checkpoint
                    value = self._parse_value()
                    array.append(value)

            self._skip_ws_comments()
            if self._peek() == ",":
                self._advance()

        if keyed and not array:
            return keyed
        if array and not keyed:
            return array
        if not keyed and not array:
            return {}

        # Mixed table fallback.
        keyed["__array__"] = array
        return keyed


def fetch_module_revision_info(base_url: str, titles: List[str]) -> Dict[str, Dict[str, Any]]:
    api_url = f"{base_url}/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions|info",
        "rvprop": "ids|timestamp|sha1|size|comment",
        "titles": "|".join(titles),
    }
    data = request_json(api_url, params=params)
    pages = data.get("query", {}).get("pages", {})
    out: Dict[str, Dict[str, Any]] = {}
    for page in pages.values():
        title = page.get("title")
        if not title:
            continue
        rev = (page.get("revisions") or [{}])[0]
        out[title] = {
            "pageid": page.get("pageid"),
            "lastrevid": page.get("lastrevid"),
            "touched": page.get("touched"),
            "revid": rev.get("revid"),
            "parentid": rev.get("parentid"),
            "timestamp": rev.get("timestamp"),
            "sha1": rev.get("sha1"),
            "size": rev.get("size"),
            "comment": rev.get("comment"),
        }
    return out


def fetch_module_raw(base_url: str, module_title: str) -> str:
    encoded = urllib.parse.quote(module_title, safe=":/")
    url = f"{base_url}/{encoded}"
    return request_text(url, params={"action": "raw"})


def batched(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def query_files_info(base_url: str, file_titles: List[str]) -> Dict[str, Dict[str, Any]]:
    api_url = f"{base_url}/api.php"
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in batched(file_titles, 50):
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|sha1|mime|size",
            "titles": "|".join(f"File:{name}" for name in chunk),
        }
        data = request_json(api_url, params=params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            if title.startswith("File:"):
                key = title_key(title[len("File:") :])
            else:
                key = title_key(title)
            exists = "missing" not in page
            info = (page.get("imageinfo") or [{}])[0] if exists else {}
            out[key] = {
                "exists": exists,
                "canonical_title": title,
                "url": info.get("url"),
                "sha1": info.get("sha1"),
                "mime": info.get("mime"),
                "size": info.get("size"),
            }
    return out


def flatten_skins(parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for champion_name, champion_data in parsed_data.items():
        if not isinstance(champion_data, dict):
            continue
        skins = champion_data.get("skins")
        if not isinstance(skins, dict):
            continue
        for skin_name, skin_data in skins.items():
            if not isinstance(skin_data, dict):
                continue
            out.append(
                {
                    "champion": champion_name,
                    "champion_id": champion_data.get("id"),
                    "skin": skin_name,
                    "skin_id": skin_data.get("id"),
                    "data": skin_data,
                }
            )
    return out


def resolve_splash_candidates(entries: List[Dict[str, Any]], base_url: str) -> None:
    needed_titles: List[str] = []
    for entry in entries:
        base = build_base_title(entry["champion"], entry["skin"])
        candidates = [
            f"{base}_HD.jpg",
            f"{base}_HD.png",
            f"{base}.jpg",
            f"{base}.png",
        ]
        entry["splash_candidates"] = candidates
        needed_titles.extend(candidates)

    needed_titles = sorted(set(needed_titles))
    info_map = query_files_info(base_url, needed_titles)

    for entry in entries:
        chosen: Optional[Dict[str, Any]] = None
        available: List[Dict[str, Any]] = []
        for candidate in entry["splash_candidates"]:
            key = title_key(candidate)
            info = info_map.get(key)
            if not info:
                continue
            item = {
                "file_name": candidate,
                "exists": info["exists"],
                "canonical_title": info["canonical_title"],
                "resolved_url": info["url"],
                "sha1": info["sha1"],
                "mime": info["mime"],
                "size": info["size"],
                "special_filepath_url": f"{base_url}/Special:FilePath/{urllib.parse.quote(candidate)}",
            }
            if info["exists"]:
                available.append(item)
                if chosen is None:
                    chosen = item

        entry["splash_resolution"] = {
            "strategy": "hd_then_regular_no_download",
            "chosen": chosen,
            "available_candidates": available,
        }


def build_grouping(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_url: Dict[str, List[Dict[str, str]]] = {}
    unresolved: List[str] = []
    entry_index = {f"{e['champion']}::{e['skin']}": e for e in entries}

    for entry in entries:
        key = f"{entry['champion']}::{entry['skin']}"
        chosen = entry["splash_resolution"]["chosen"]
        if not chosen or not chosen.get("resolved_url"):
            unresolved.append(key)
            entry["target_folder"] = "_unresolved"
            continue

        url = chosen["resolved_url"]
        by_url.setdefault(url, []).append(
            {
                "key": key,
                "champion": entry["champion"],
                "skin": entry["skin"],
            }
        )

    grouped = []
    for url, refs in by_url.items():
        champions = sorted({r["champion"] for r in refs})
        is_shared = len(champions) > 1
        group_id = stable_hash(url)
        folder = f"shared/{group_id}" if is_shared else safe_filename(champions[0])
        for ref in refs:
            entry_key = ref["key"]
            if entry_key in entry_index:
                entry_index[entry_key]["target_folder"] = folder
        grouped.append(
            {
                "group_id": group_id,
                "resolved_url": url,
                "champions": champions,
                "entry_count": len(refs),
                "is_shared_multi_champion": is_shared,
                "folder": folder,
                "entries": refs,
            }
        )

    grouped.sort(key=lambda x: (not x["is_shared_multi_champion"], x["folder"], x["resolved_url"]))
    return {
        "groups": grouped,
        "unresolved_entries": sorted(unresolved),
    }


def load_previous_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def entry_snapshot(entry: Dict[str, Any]) -> Dict[str, Any]:
    # Stable subset used for diffs across runs.
    return {
        "champion_id": entry["champion_id"],
        "skin_id": entry["skin_id"],
        "data": entry["data"],
        "chosen_url": (entry["splash_resolution"]["chosen"] or {}).get("resolved_url"),
        "chosen_file_name": (entry["splash_resolution"]["chosen"] or {}).get("file_name"),
        "chosen_sha1": (entry["splash_resolution"]["chosen"] or {}).get("sha1"),
        "target_folder": entry.get("target_folder"),
    }


def build_diff(previous: Optional[Dict[str, Any]], current_entries: List[Dict[str, Any]], source_info: Dict[str, Any]) -> Dict[str, Any]:
    if not previous:
        added_keys = sorted(f"{e['champion']}::{e['skin']}" for e in current_entries)
        return {
            "has_previous_local_copy": False,
            "source_changed": True,
            "added_entries": len(current_entries),
            "removed_entries": 0,
            "changed_entries": 0,
            "added_keys_sample": added_keys[:100],
            "removed_keys_sample": [],
            "changed_keys_sample": [],
        }

    prev_entries = previous.get("entries", [])
    prev_map = {f"{e['champion']}::{e['skin']}": e for e in prev_entries if "champion" in e and "skin" in e}
    curr_map = {f"{e['champion']}::{e['skin']}": e for e in current_entries}

    prev_source = previous.get("source", {}).get("modules", {})
    source_changed = False
    for title, info in source_info.get("modules", {}).items():
        prev_rev = (prev_source.get(title) or {}).get("revid")
        if prev_rev != info.get("revid"):
            source_changed = True
            break

    added = sorted(set(curr_map) - set(prev_map))
    removed = sorted(set(prev_map) - set(curr_map))
    changed: List[str] = []
    for key in sorted(set(curr_map) & set(prev_map)):
        if entry_snapshot(curr_map[key]) != entry_snapshot(prev_map[key]):
            changed.append(key)

    return {
        "has_previous_local_copy": True,
        "previous_generated_at_utc": previous.get("generated_at_utc"),
        "source_changed": source_changed,
        "added_entries": len(added),
        "removed_entries": len(removed),
        "changed_entries": len(changed),
        "added_keys_sample": added[:100],
        "removed_keys_sample": removed[:100],
        "changed_keys_sample": changed[:100],
    }


def parse_release_date(value: Any) -> Optional[dt.date]:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def choose_oldest_entry(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sort_key(entry: Dict[str, Any]) -> Any:
        release = parse_release_date(entry.get("data", {}).get("release"))
        has_release = release is not None
        # Unknown release goes last.
        return (
            0 if has_release else 1,
            release or dt.date.max,
            entry.get("champion", ""),
            entry.get("skin", ""),
        )

    return sorted(entries, key=sort_key)[0]


def build_artifact_records(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        chosen = entry.get("splash_resolution", {}).get("chosen")
        url = (chosen or {}).get("resolved_url")
        if not url:
            continue
        by_url.setdefault(url, []).append(entry)

    artifacts: List[Dict[str, Any]] = []
    for url, group_entries in by_url.items():
        champions = sorted({e["champion"] for e in group_entries})
        is_true_group = len(champions) > 1
        canonical = choose_oldest_entry(group_entries)
        canonical_chosen = canonical["splash_resolution"]["chosen"] or {}
        file_name = canonical_chosen.get("file_name") or f"{stable_hash(url)}.jpg"
        group_id = stable_hash(url)
        folder = "shared" if is_true_group else safe_filename(canonical["champion"])

        artifacts.append(
            {
                "artifact_id": group_id,
                "resolved_url": url,
                "is_shared_multi_champion": is_true_group,
                "champions": champions,
                "entry_count": len(group_entries),
                "folder": folder,
                "canonical_entry": {
                    "key": f"{canonical['champion']}::{canonical['skin']}",
                    "champion": canonical["champion"],
                    "skin": canonical["skin"],
                    "release": canonical.get("data", {}).get("release"),
                    "splashartist": canonical.get("data", {}).get("splashartist"),
                    "skin_data": canonical.get("data", {}),
                },
                "file_name": file_name,
                "file_sha1": canonical_chosen.get("sha1"),
                "file_mime": canonical_chosen.get("mime"),
                "file_size": canonical_chosen.get("size"),
                "entries": [
                    {
                        "key": f"{e['champion']}::{e['skin']}",
                        "champion": e["champion"],
                        "skin": e["skin"],
                        "release": e.get("data", {}).get("release"),
                        "skin_data": e.get("data", {}),
                    }
                    for e in sorted(group_entries, key=lambda x: (x["champion"], x["skin"]))
                ],
            }
        )

    artifacts.sort(key=lambda x: (not x["is_shared_multi_champion"], x["folder"], x["file_name"]))
    return artifacts


def write_url_shortcut(path: Path, url: str) -> None:
    content = "[InternetShortcut]\nURL=" + url + "\n"
    path.write_text(content, encoding="utf-8")


class RateLimiter:
    def __init__(self, max_per_second: float) -> None:
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second if max_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                sleep_for = self.next_allowed - now
                time.sleep(sleep_for)
                now = time.monotonic()
            self.next_allowed = now + self.min_interval


def maybe_skip_existing(image_path: Path, force_redownload: bool) -> bool:
    if force_redownload:
        return False
    if not image_path.exists():
        return False
    return True


def detect_image_format(data: bytes) -> Optional[str]:
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def classify_download_error(error_text: str) -> str:
    text = (error_text or "").lower()
    if "429" in text:
        return "rate_limited"
    if "404" in text:
        return "not_found"
    if "403" in text:
        return "forbidden"
    if "5" in text and "http" in text:
        return "server_error"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "connection" in text:
        return "connection"
    if "invalid image content" in text:
        return "invalid_image"
    if "size mismatch" in text:
        return "size_mismatch"
    return "other"


def ensure_unique_output_name(
    folder_name: str,
    requested_name: str,
    folder_name_registry: Dict[str, Dict[str, str]],
    artifact_id: str,
) -> str:
    requested_name = safe_filename(requested_name)
    folder_registry = folder_name_registry.setdefault(folder_name, {})
    current_owner = folder_registry.get(requested_name)
    if current_owner is None or current_owner == artifact_id:
        folder_registry[requested_name] = artifact_id
        return requested_name

    stem, suffix = Path(requested_name).stem, Path(requested_name).suffix
    candidate = f"{stem}__{artifact_id[:8]}{suffix}"
    idx = 2
    while candidate in folder_registry and folder_registry[candidate] != artifact_id:
        candidate = f"{stem}__{artifact_id[:8]}_{idx}{suffix}"
        idx += 1
    folder_registry[candidate] = artifact_id
    return candidate


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def list_to_pipe(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "|".join(str(x) for x in value)


def to_int_or_zero(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_hf_bundle(
    base_dir: Path,
    artifacts: List[Dict[str, Any]],
    metadata_generated_at: str,
    source_modules: Dict[str, Any],
) -> Dict[str, Any]:
    hf_root = base_dir / "hf"
    hf_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for artifact in artifacts:
        canonical = artifact.get("canonical_entry", {})
        canonical_data = canonical.get("skin_data", {}) if isinstance(canonical.get("skin_data"), dict) else {}
        for entry in artifact.get("entries", []):
            skin_data = entry.get("skin_data", {}) if isinstance(entry.get("skin_data"), dict) else {}
            rows.append(
                {
                    "artifact_id": artifact.get("artifact_id", ""),
                    "resolved_url": artifact.get("resolved_url", ""),
                    "is_shared_multi_champion": bool(artifact.get("is_shared_multi_champion", False)),
                    "folder": artifact.get("folder", ""),
                    "output_file_name": artifact.get("output_file_name", artifact.get("file_name", "")),
                    "entry_count": int(artifact.get("entry_count", 0)),
                    "file_sha1": normalize_string(artifact.get("file_sha1")),
                    "file_mime": normalize_string(artifact.get("file_mime")),
                    "file_size": to_int_or_zero(artifact.get("file_size")),
                    "canonical_key": canonical.get("key", ""),
                    "canonical_champion": canonical.get("champion", ""),
                    "canonical_skin": canonical.get("skin", ""),
                    "canonical_release": normalize_string(canonical.get("release")),
                    "canonical_set": list_to_pipe(canonical_data.get("set")),
                    "canonical_splashartist": list_to_pipe(canonical_data.get("splashartist")),
                    "entry_key": entry.get("key", ""),
                    "champion": entry.get("champion", ""),
                    "skin": entry.get("skin", ""),
                    "release": normalize_string(entry.get("release")),
                    "set": list_to_pipe(skin_data.get("set")),
                    "splashartist": list_to_pipe(skin_data.get("splashartist")),
                    "availability": normalize_string(skin_data.get("availability")),
                    "cost": to_int_or_zero(skin_data.get("cost")),
                    "generated_at_utc": metadata_generated_at,
                    "source_skindata_revid": to_int_or_zero((source_modules.get("Module:SkinData/data") or {}).get("revid")),
                }
            )

    jsonl_path = hf_root / "records.jsonl"
    csv_path = hf_root / "records.csv"
    parquet_path = hf_root / "records.parquet"
    write_jsonl(jsonl_path, rows)

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    parquet_written = False
    parquet_error = ""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)
        parquet_written = True
    except Exception as exc:  # noqa: BLE001
        parquet_error = str(exc)

    data_file = "hf/records.parquet" if parquet_written else "hf/records.jsonl"
    source_repo_url = "https://github.com/esbe1175/lol-splash-collector"
    skindata_rev = (source_modules.get("Module:SkinData/data") or {}).get("revid")
    filename_rev = (source_modules.get("Module:Filename") or {}).get("revid")
    card_text = (
        "---\n"
        "configs:\n"
        "  - config_name: default\n"
        "    data_files:\n"
        "      - split: train\n"
        f"        path: {data_file}\n"
        "---\n\n"
        "# lol-splash-collector dataset export\n\n"
        f"Up to date as of **{metadata_generated_at}** (UTC).\n\n"
        f"Generated with [esbe1175/lol-splash-collector]({source_repo_url}).\n\n"
        "Source wiki module revisions used:\n\n"
        f"- `Module:SkinData/data`: `{skindata_rev}`\n"
        f"- `Module:Filename`: `{filename_rev}`\n\n"
        "- `hf/records.jsonl`: line-delimited records for each champion-skin row.\n"
        "- `hf/records.csv`: tabular export for spreadsheet usage.\n"
        "- `hf/records.parquet`: parquet export (written when pyarrow is available).\n"
    )
    (hf_root / "README.md").write_text(card_text, encoding="utf-8")
    (base_dir / "README.md").write_text(card_text, encoding="utf-8")

    summary = {
        "hf_root": str(hf_root),
        "row_count": len(rows),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path) if parquet_written else None,
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
        "card_path": str(hf_root / "README.md"),
        "root_card_path": str(base_dir / "README.md"),
    }
    (hf_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def download_one_file(
    rate_limiter: RateLimiter,
    url: str,
    target_path: Path,
    timeout_seconds: int,
    retries: int,
    retry_backoff_seconds: float,
    expected_size: Optional[int],
    strict_size_check: bool,
) -> Dict[str, Any]:
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        try:
            rate_limiter.wait()
            with requests.get(
                url,
                stream=True,
                timeout=timeout_seconds,
                headers={"User-Agent": f"lol-splash-collector/{TOOL_VERSION}"},
            ) as response:
                response.raise_for_status()
                first_chunk: Optional[bytes] = None
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            if first_chunk is None:
                                first_chunk = chunk
                            handle.write(chunk)

            downloaded_size = tmp_path.stat().st_size
            if first_chunk is None:
                first_chunk = b""

            detected_format = detect_image_format(first_chunk)
            if detected_format is None:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError("invalid image content (unknown signature)")

            if strict_size_check and expected_size is not None and downloaded_size != expected_size:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"size mismatch (expected {expected_size}, got {downloaded_size})"
                )

            tmp_path.replace(target_path)
            size_mismatch = expected_size is not None and downloaded_size != expected_size
            return {
                "ok": True,
                "status": "downloaded_size_mismatch" if size_mismatch else "downloaded",
                "size": target_path.stat().st_size,
                "attempts": attempt,
                "error": None,
                "error_type": None,
                "detected_format": detected_format,
                "size_mismatch": size_mismatch,
                "expected_size": expected_size,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))

    return {
        "ok": False,
        "status": "failed",
        "size": None,
        "attempts": retries,
        "error": last_error,
        "error_type": classify_download_error(last_error or ""),
        "detected_format": None,
        "size_mismatch": False,
        "expected_size": expected_size,
    }


def materialize_assets(
    output_dir: Path,
    artifacts: List[Dict[str, Any]],
    source_modules: Dict[str, Any],
    dry_run: bool,
    download_workers: int,
    max_requests_per_second: float,
    download_timeout_seconds: int,
    download_retries: int,
    retry_backoff_seconds: float,
    force_redownload: bool,
    max_downloads: int,
    strict_size_check: bool,
) -> Dict[str, Any]:
    dataset_root = output_dir / ("dataset_dryrun" if dry_run else "dataset")
    dataset_root.mkdir(parents=True, exist_ok=True)

    created_folders = 0
    wrote_shortcuts = 0
    wrote_metadata = 0
    local_image_exists = 0
    local_image_missing = 0
    would_download = 0
    downloaded_count = 0
    skipped_existing_count = 0
    failed_count = 0
    downloaded_with_size_mismatch_count = 0
    failure_by_type: Dict[str, int] = {}
    folder_name_registry: Dict[str, Dict[str, str]] = {}

    tasks: List[Dict[str, Any]] = []

    for item in artifacts:
        folder_path = dataset_root / item["folder"]
        if not folder_path.exists():
            created_folders += 1
        folder_path.mkdir(parents=True, exist_ok=True)

        image_file_name = ensure_unique_output_name(
            folder_name=item["folder"],
            requested_name=item["file_name"],
            folder_name_registry=folder_name_registry,
            artifact_id=item["artifact_id"],
        )
        item["output_file_name"] = image_file_name
        image_path = folder_path / image_file_name
        image_exists = image_path.exists()
        if image_exists:
            local_image_exists += 1
        else:
            local_image_missing += 1
            would_download += 1

        shortcut_path = folder_path / f"{image_file_name}.url"
        metadata_path = folder_path / f"{image_file_name}.metadata.json"

        item["materialization"] = {
            "image_path": str(image_path),
            "shortcut_path": str(shortcut_path) if dry_run else None,
            "metadata_path": str(metadata_path),
            "download_status": "pending",
            "error": "",
            "error_type": "",
            "attempts": 0,
            "detected_format": "",
            "size_mismatch": False,
            "expected_size": item.get("file_size"),
        }

        if dry_run:
            write_url_shortcut(shortcut_path, item["resolved_url"])
            wrote_shortcuts += 1
            item["materialization"]["download_status"] = "not_downloaded_dry_run"
            continue

        expected_size = item.get("file_size")
        if maybe_skip_existing(image_path, force_redownload):
            skipped_existing_count += 1
            item["materialization"]["download_status"] = "skipped_exists"
            item["materialization"]["attempts"] = 0
            continue

        tasks.append(
            {
                "url": item["resolved_url"],
                "image_path": image_path,
                "expected_size": expected_size,
                "artifact": item,
            }
        )

    if not dry_run and max_downloads > 0:
        tasks = tasks[:max_downloads]

    if not dry_run and tasks:
        rate_limiter = RateLimiter(max_requests_per_second)

        start = time.monotonic()
        total = len(tasks)
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as pool:
            future_map = {}
            for task in tasks:
                future = pool.submit(
                    download_one_file,
                    rate_limiter,
                    task["url"],
                    task["image_path"],
                    download_timeout_seconds,
                    download_retries,
                    retry_backoff_seconds,
                    task["expected_size"],
                    strict_size_check,
                )
                future_map[future] = task

            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                item = task["artifact"]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "ok": False,
                        "status": "failed",
                        "error": str(exc),
                        "attempts": download_retries,
                    }

                status = result["status"]
                item["materialization"]["download_status"] = status
                item["materialization"]["error"] = normalize_string(result.get("error"))
                item["materialization"]["error_type"] = normalize_string(result.get("error_type"))
                item["materialization"]["attempts"] = result.get("attempts", 0)
                item["materialization"]["detected_format"] = normalize_string(result.get("detected_format"))
                item["materialization"]["size_mismatch"] = result.get("size_mismatch", False)

                if status in ("downloaded", "downloaded_size_mismatch"):
                    downloaded_count += 1
                    if status == "downloaded_size_mismatch":
                        downloaded_with_size_mismatch_count += 1
                elif status == "failed":
                    failed_count += 1
                    et = result.get("error_type") or "other"
                    failure_by_type[et] = failure_by_type.get(et, 0) + 1

                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else 0.0
                print(
                    "Download progress: "
                    f"{completed}/{total} "
                    f"(downloaded={downloaded_count}, failed={failed_count}) "
                    f"rate={rate:.2f}/s eta={eta:.1f}s"
                )

    for item in artifacts:
        image_file_name = safe_filename(item["file_name"])
        metadata_path = Path(item["materialization"]["metadata_path"])
        image_path = Path(item["materialization"]["image_path"])
        payload = {
            "dry_run": dry_run,
            "resolved_url": item["resolved_url"],
            "image_file_name": image_file_name,
            "is_shared_multi_champion": item["is_shared_multi_champion"],
            "folder": item["folder"],
            "canonical_entry": item["canonical_entry"],
            "entry_count": item["entry_count"],
            "entries": item["entries"],
            "local_image_state": {
                "expected_path": str(image_path),
                "exists": image_path.exists(),
            },
            "source_modules": source_modules,
            "download_status": item["materialization"]["download_status"],
            "download_attempts": item["materialization"]["attempts"],
            "download_error": item["materialization"]["error"],
            "download_error_type": item["materialization"]["error_type"],
            "download_detected_format": item["materialization"]["detected_format"],
            "download_size_mismatch_vs_api": item["materialization"]["size_mismatch"],
            "download_expected_size_from_api": item["materialization"]["expected_size"],
        }
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        wrote_metadata += 1

    return {
        "dry_run": dry_run,
        "dataset_root": str(dataset_root),
        "artifact_count": len(artifacts),
        "created_folders": created_folders,
        "wrote_shortcuts": wrote_shortcuts,
        "wrote_metadata_files": wrote_metadata,
        "local_image_exists": local_image_exists,
        "local_image_missing": local_image_missing,
        "would_download_count": would_download,
        "downloaded_count": downloaded_count,
        "downloaded_with_size_mismatch_count": downloaded_with_size_mismatch_count,
        "skipped_existing_count": skipped_existing_count,
        "failed_count": failed_count,
        "failure_by_type": failure_by_type,
        "download_settings": {
            "workers": download_workers,
            "max_requests_per_second": max_requests_per_second,
            "timeout_seconds": download_timeout_seconds,
            "retries": download_retries,
            "retry_backoff_seconds": retry_backoff_seconds,
            "force_redownload": force_redownload,
            "max_downloads": max_downloads,
            "strict_size_check": strict_size_check,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LoL splash metadata database.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Wiki base URL")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--metadata-name", default="metadata.json", help="Metadata file name")
    parser.add_argument(
        "--materialize-assets",
        action="store_true",
        help="Create dataset folders and per-image metadata/shortcuts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run for asset materialization (creates .url shortcuts + metadata, no image download).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (used when not --dry-run).",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=float,
        default=2.0,
        help="Global cap for HTTP request rate during downloads.",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=int,
        default=60,
        help="Per-request timeout for downloads.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=4,
        help="Retry attempts per file download.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial backoff for download retries (doubles each retry).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download even when local file appears valid.",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=0,
        help="Optional cap for number of files to download this run (0 means no cap).",
    )
    parser.add_argument(
        "--strict-size-check",
        action="store_true",
        help="Treat API file size mismatch as hard failure (disabled by default).",
    )
    parser.add_argument(
        "--skip-hf-export",
        action="store_true",
        help="Skip writing Hugging Face-friendly tabular exports under output/hf.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / args.metadata_name

    previous = load_previous_metadata(metadata_path)
    previous_mtime_utc = None
    if metadata_path.exists():
        previous_mtime_utc = dt.datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=dt.timezone.utc).isoformat()
    created_backup_path = backup_metadata_file(metadata_path, output_dir)

    source_modules = fetch_module_revision_info(base_url, [SKINDATA_MODULE, FILENAME_MODULE])
    skin_data_raw = fetch_module_raw(base_url, SKINDATA_MODULE)

    parser_instance = LuaTableParser(skin_data_raw)
    parsed = parser_instance.parse()
    if not isinstance(parsed, dict):
        raise RuntimeError("Parsed skin data is not a dictionary table.")

    entries = flatten_skins(parsed)
    resolve_splash_candidates(entries, base_url)
    grouping = build_grouping(entries)
    diff = build_diff(previous, entries, {"modules": source_modules})

    grouped = grouping["groups"]
    unresolved = grouping["unresolved_entries"]
    shared_groups = [g for g in grouped if g["is_shared_multi_champion"]]
    artifacts = build_artifact_records(entries)

    materialization_summary: Optional[Dict[str, Any]] = None
    if args.materialize_assets:
        if args.dry_run:
            materialization_summary = materialize_assets(
                output_dir=output_dir,
                artifacts=artifacts,
                source_modules=source_modules,
                dry_run=True,
                download_workers=max(1, args.download_workers),
                max_requests_per_second=max(0.1, args.max_requests_per_second),
                download_timeout_seconds=max(1, args.download_timeout_seconds),
                download_retries=max(1, args.download_retries),
                retry_backoff_seconds=max(0.1, args.retry_backoff_seconds),
                force_redownload=args.force_redownload,
                max_downloads=max(0, args.max_downloads),
                strict_size_check=args.strict_size_check,
            )
        else:
            materialization_summary = materialize_assets(
                output_dir=output_dir,
                artifacts=artifacts,
                source_modules=source_modules,
                dry_run=False,
                download_workers=max(1, args.download_workers),
                max_requests_per_second=max(0.1, args.max_requests_per_second),
                download_timeout_seconds=max(1, args.download_timeout_seconds),
                download_retries=max(1, args.download_retries),
                retry_backoff_seconds=max(0.1, args.retry_backoff_seconds),
                force_redownload=args.force_redownload,
                max_downloads=max(0, args.max_downloads),
                strict_size_check=args.strict_size_check,
            )

    metadata = {
        "tool": {
            "name": "lol-splash-collector",
            "version": TOOL_VERSION,
        },
        "generated_at_utc": utc_now_iso(),
        "local_copy": {
            "path": str(metadata_path),
            "previous_exists": previous is not None,
            "previous_file_modified_at_utc": previous_mtime_utc,
            "previous_generated_at_utc": (previous or {}).get("generated_at_utc"),
            "backup_created": created_backup_path is not None,
            "backup_path": str(created_backup_path) if created_backup_path else None,
        },
        "source": {
            "wiki_base_url": base_url,
            "modules": source_modules,
            "notes": "Image links resolved via MediaWiki imageinfo URL, without downloading files.",
        },
        "resolution_policy": {
            "download_images": False,
            "candidate_order": ["_HD.jpg", "_HD.png", ".jpg", ".png"],
            "match_mode": "resolved_url",
        },
        "stats": {
            "champion_count": len(parsed),
            "skin_entry_count": len(entries),
            "resolved_entry_count": len(entries) - len(unresolved),
            "unresolved_entry_count": len(unresolved),
            "resolved_url_group_count": len(grouped),
            "shared_multi_champion_group_count": len(shared_groups),
            "artifact_count": len(artifacts),
        },
        "diff_from_previous": diff,
        "groups": grouped,
        "artifacts": artifacts,
        "unresolved_entries": unresolved,
        "entries": sorted(entries, key=lambda x: (x["champion"], x["skin"])),
    }
    if materialization_summary is not None:
        metadata["materialization"] = materialization_summary

    hf_export_summary = None
    if not args.skip_hf_export:
        hf_base_dir = output_dir
        if materialization_summary is not None:
            hf_base_dir = Path(materialization_summary["dataset_root"])
        hf_export_summary = export_hf_bundle(
            base_dir=hf_base_dir,
            artifacts=artifacts,
            metadata_generated_at=metadata["generated_at_utc"],
            source_modules=source_modules,
        )
        metadata["hf_export"] = hf_export_summary

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Entries: {metadata['stats']['skin_entry_count']}")
    print(f"Resolved: {metadata['stats']['resolved_entry_count']}")
    print(f"Shared multi-champion groups: {metadata['stats']['shared_multi_champion_group_count']}")
    print(f"Unresolved: {metadata['stats']['unresolved_entry_count']}")
    if created_backup_path is not None:
        print(f"Backed up previous metadata to: {created_backup_path}")
    if materialization_summary is not None and materialization_summary["dry_run"]:
        print(f"Dry-run dataset root: {materialization_summary['dataset_root']}")
        print(f"Wrote shortcuts: {materialization_summary['wrote_shortcuts']}")
        print(f"Wrote per-art metadata files: {materialization_summary['wrote_metadata_files']}")
    if materialization_summary is not None and not materialization_summary["dry_run"]:
        print(f"Dataset root: {materialization_summary['dataset_root']}")
        print(f"Downloaded: {materialization_summary['downloaded_count']}")
        print(f"Downloaded with API size mismatch: {materialization_summary['downloaded_with_size_mismatch_count']}")
        print(f"Skipped existing: {materialization_summary['skipped_existing_count']}")
        print(f"Failed: {materialization_summary['failed_count']}")
        if materialization_summary["failure_by_type"]:
            print(f"Failure breakdown: {materialization_summary['failure_by_type']}")
    if hf_export_summary is not None:
        print(f"HF export rows: {hf_export_summary['row_count']}")
        if hf_export_summary["parquet_written"]:
            print(f"HF parquet: {hf_export_summary['parquet_path']}")
        else:
            print("HF parquet: not written (pyarrow unavailable)")


if __name__ == "__main__":
    main()
