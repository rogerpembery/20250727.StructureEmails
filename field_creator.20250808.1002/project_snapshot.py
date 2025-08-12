#!/usr/bin/env python3
"""
JSON exporter for project structure + appendices (for GPT coding agents)

Upgrades:
- Adds `exclusions`: [{path, rel_path, reason}] explaining why a file wasn't appended.
- Each `files` entry includes `rel_path` and `language`.
- Each appendix entry includes `sha256` and `encoding`.
- Meta echoes `large_threshold_kb` (as well as bytes).
- Always excludes junk dirs (venv/.venv/venv*, .git, node_modules, __pycache__, etc.).
- .env-like files are included (no redaction).
- Appendices include TEXT files only, excluding .json and .log (by extension).
- "Large" prompt threshold is set in **kB** via --large-kb (bytes fallback via --max-size).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

# ---- Config ----

APPENDIX_EXCLUDE_EXTS = {".json", ".log"}  # excluded from appendices; still listed in structure
DEFAULT_MAX_SIZE = 5 * 1024  # 5 KiB (bytes), overridden by --large-kb

ALWAYS_EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".eggs", ".cache", ".idea", ".vscode",
}

EXT_TO_LANG = {
    ".py":"python",".js":"javascript",".ts":"typescript",".tsx":"tsx",".jsx":"jsx",
    ".html":"html",".htm":"html",".css":"css",".scss":"scss",".md":"markdown",
    ".yml":"yaml",".yaml":"yaml",".toml":"toml",".ini":"ini",".cfg":"ini",
    ".sh":"bash",".zsh":"zsh",".bat":"bat",".ps1":"powershell",".sql":"sql",
    ".xml":"xml",".csv":"csv",".env":"","txt":"", ".txt":""
}

_VENV_NAME_RE = re.compile(r"^\.?venv.*$", re.IGNORECASE)

def is_virtualenv_dir_name(name: str) -> bool:
    return bool(_VENV_NAME_RE.match(name))

def is_junk_dir_name(name: str) -> bool:
    lname = name.lower()
    return lname in ALWAYS_EXCLUDE_DIRS or is_virtualenv_dir_name(name)

# ---- Helpers ----

def is_probably_text(path: Path, sample_bytes: int = 8192) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_bytes)
        if b"\x00" in chunk:
            return False
        ctrl = sum(b < 9 or (13 < b < 32) for b in chunk)
        if chunk and (ctrl / len(chunk)) > 0.05:
            return False
        try:
            chunk.decode("utf-8", errors="strict")
            return True
        except UnicodeDecodeError:
            try:
                chunk.decode("latin-1")
                return True
            except UnicodeDecodeError:
                return False
    except Exception:
        return False

def relpath_sorted_children(root: Path) -> Iterable[Path]:
    try:
        entries = list(root.iterdir())
    except PermissionError:
        return []
    filtered = []
    for p in entries:
        try:
            if p.is_dir() and is_junk_dir_name(p.name):
                continue
        except PermissionError:
            pass
        filtered.append(p)
    filtered.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    return filtered

def human_size(n: int) -> str:
    orig = n
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{orig} B" if unit=="B" else f"{orig/1024:.1f} {unit}"
        orig = n
        n /= 1024.0

def infer_lang(ext: str) -> str:
    return EXT_TO_LANG.get(ext.lower(), "")

def read_bytes(path: Path) -> Optional[bytes]:
    try:
        with path.open("rb") as f:
            return f.read()
    except Exception:
        return None

def decode_text(b: bytes) -> Tuple[str, str]:
    """
    Return (text, encoding). Try utf-8 strict, then latin-1 strict, then utf-8 replace.
    """
    try:
        return b.decode("utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        try:
            return b.decode("latin-1", errors="strict"), "latin-1"
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="replace"), "utf-8 (replace)"

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---- Core traversal ----

def build_tree_and_flat(
    root: Path,
    follow_symlinks: bool = False,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]], int, int]:
    dir_count = 0
    file_count = 0

    def recurse_dir(d: Path) -> Dict[str, Any]:
        nonlocal dir_count, file_count
        node = {"type":"directory","name":d.name,"path":str(d.resolve()),"children":[]}
        dir_count += 1
        for child in relpath_sorted_children(d):
            try:
                if child.is_dir():
                    if child.is_symlink() and not follow_symlinks:
                        node["children"].append({
                            "type":"symlink_dir","name":child.name,
                            "path":str(child.resolve()) if child.exists() else str(child),
                        })
                        continue
                    node["children"].append(recurse_dir(child))
                elif child.is_file():
                    try:
                        size = child.stat().st_size
                    except Exception:
                        size = None
                    node["children"].append({
                        "type":"file","name":child.name,
                        "path":str(child.resolve()),
                        "size_bytes":size,
                    })
                    file_count += 1
                else:
                    node["children"].append({"type":"special","name":child.name,"path":str(child)})
            except PermissionError:
                node["children"].append({"type":"error","name":child.name,"path":str(child),"error":"permission_denied"})
            except FileNotFoundError:
                node["children"].append({"type":"error","name":child.name,"path":str(child),"error":"not_found"})
        return node

    tree = recurse_dir(root)

    flat_files: list[Dict[str, Any]] = []
    def collect_files(node: Dict[str, Any]):
        if node.get("type") == "file":
            flat_files.append(node)
            return
        for c in node.get("children", []):
            collect_files(c)
    collect_files(tree)
    return tree, flat_files, dir_count, file_count

# ---- Appendix decision (with reason) ----

def decide_appendix(path: Path, max_size: int, mode: str, out_path: Path) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """
    Returns (include, reason_if_excluded, raw_bytes_if_included_or_needed).
    Reasons: 'self_output', 'excluded_ext', 'binary', 'large_skipped', 'other'
    """
    # Skip the output file itself
    if path.resolve() == out_path.resolve():
        return (False, "self_output", None)

    ext = path.suffix.lower()
    if ext in APPENDIX_EXCLUDE_EXTS:
        return (False, "excluded_ext", None)

    if not is_probably_text(path):
        return (False, "binary", None)

    try:
        size = path.stat().st_size
    except Exception:
        return (False, "other", None)

    if size > max_size:
        if mode == "no":
            return (False, "large_skipped", None)
        elif mode == "ask":
            rel = path.relative_to(Path.cwd())
            while True:
                resp = input(
                    f"File '{rel}' is {human_size(size)} (> {max_size} bytes). "
                    "Include in appendix? [y/N/a=all yes/n=all no] "
                ).strip().lower()
                if resp in ("y","yes"):
                    break
                if resp in ("n","no",""):
                    return (False, "large_skipped", None)
                if resp in ("a","all","always"):
                    # Treat as yes for this file; (global mode change not persisted—intentional)
                    break
                if resp in ("never","nn"):
                    return (False, "large_skipped", None)
                print("Please answer y, n, a (all yes), or 'never' (all no).")

    b = read_bytes(path)
    if b is None:
        return (False, "other", None)
    return (True, None, b)

# ---- Output ----

def write_report_json(
    root: Path,
    out_path: Path,
    follow_symlinks: bool,
    max_size: int,
    large_mode: str,
):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tree, flat_files, dir_count, file_count = build_tree_and_flat(root, follow_symlinks=follow_symlinks)

    appendices = []
    exclusions = []
    total_appendix_bytes = 0

    files_sorted = sorted(flat_files, key=lambda f: f["path"].lower())

    for f in files_sorted:
        p = Path(f["path"])
        if not p.is_file():
            continue

        include, reason, raw = decide_appendix(p, max_size=max_size, mode=large_mode, out_path=out_path)
        rel = str(p.relative_to(root))

        if include and raw is not None:
            text, encoding = decode_text(raw)
            size = None
            try:
                size = p.stat().st_size
            except Exception:
                pass
            lang = infer_lang(p.suffix.lower())
            appendices.append({
                "path": rel,
                "abs_path": str(p.resolve()),
                "size_bytes": size,
                "language": lang,
                "encoding": encoding,
                "sha256": sha256_hex(raw),
                "content": text,
            })
            if size:
                total_appendix_bytes += size
        else:
            exclusions.append({
                "path": str(p.resolve()),
                "rel_path": rel,
                "reason": reason or "other",
            })

    # Enrich `files` entries with rel_path and language
    enriched_files = []
    for f in files_sorted:
        p = Path(f["path"])
        enriched = dict(f)
        try:
            enriched["rel_path"] = str(p.relative_to(root))
        except Exception:
            enriched["rel_path"] = f["path"]
        enriched["language"] = infer_lang(p.suffix.lower())
        enriched_files.append(enriched)

    data = {
        "meta": {
            "generated": timestamp,
            "root": str(root.resolve()),
            "large_threshold_bytes": max_size,
            "large_threshold_kb": round(max_size / 1024, 2),
            "large_mode": large_mode,  # 'ask' | 'yes' | 'no'
            "excluded_directory_rules": {
                "virtualenv_regex": r"^\.?venv.*$",
                "name_set": sorted(list(ALWAYS_EXCLUDE_DIRS)),
            },
            "summary": {
                "directories": dir_count,
                "files": file_count,
                "appendices": len(appendices),
                "appendices_total_bytes": total_appendix_bytes,
                "exclusions": len(exclusions),
            },
        },
        "structure": tree,          # nested directory tree
        "files": enriched_files,    # flat list with rel_path + language
        "appendices": appendices,   # included file contents + sha256 + encoding
        "exclusions": exclusions,   # non-appended files with reasons
    }

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON report to: {out_path}")

# ---- CLI ----

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export a recursive project structure and text-file appendices as JSON for GPT agents."
    )
    p.add_argument("-o","--output", default="PROJECT_STRUCTURE_AND_APPENDICES.json",
                   help="Output JSON filename (default: %(default)s)")
    p.add_argument("--large-kb", type=int, default=None,
                   help="Size threshold in kilobytes for prompting on large files.")
    p.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE,
                   help=f"Fallback threshold in bytes if --large-kb not provided (default: {DEFAULT_MAX_SIZE}).")
    p.add_argument("--follow-symlinks", action="store_true",
                   help="Follow symlinked directories (off by default).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--yes-large", action="store_true",
                       help="Automatically include large files without prompting.")
    group.add_argument("--no-large", action="store_true",
                       help="Automatically skip large files without prompting.")
    return p.parse_args(argv)

def main():
    args = parse_args()
    root = Path.cwd()
    out_path = (root / args.output).resolve()

    large_mode = "ask"
    if args.yes_large:
        large_mode = "yes"
    elif args.no_large:
        large_mode = "no"

    max_size = args.max_size
    if args.large_kb is not None:
        max_size = int(args.large_kb) * 1024  # kB -> bytes

    try:
        write_report_json(
            root=root,
            out_path=out_path,
            follow_symlinks=args.follow_symlinks,
            max_size=max_size,
            large_mode=large_mode,
        )
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()
