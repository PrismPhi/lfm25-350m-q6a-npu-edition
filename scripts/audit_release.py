#!/usr/bin/env python3
"""Audit the public repository and optional model-release staging directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


MAX_REPO_BYTES = 5 * 1024 * 1024
REPO_BINARY_SUFFIXES = {
    ".bin", ".dll", ".dylib", ".gguf", ".key", ".npy", ".npz", ".onnx",
    ".p12", ".pem", ".safetensors", ".so",
}
MODEL_FORBIDDEN_SUFFIXES = {
    ".bin", ".dll", ".dylib", ".gguf", ".key", ".p12", ".pem", ".so",
}
FORBIDDEN_DIRS = {
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "venv",
}
TEXT_SUFFIXES = {
    "", ".cfg", ".env", ".example", ".ini", ".jinja", ".json", ".md", ".py",
    ".sh", ".toml", ".txt", ".yaml", ".yml",
}
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")
TICK = chr(96)
INLINE_CODE = re.compile(TICK + r"([^" + TICK + r"\r\n]+)" + TICK)
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:>=|<=|>|<|~|\u2248|\u2264|\u2265)?"
    r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"(?:/\d+(?:\.\d+)?)?"
    r"(?![A-Za-z0-9_])"
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
TABLE_DELIMITER = re.compile(r"^:?-{3,}:?$")
HF_MODEL_CARD_FRONT_MATTER = [
    "license: other",
    "license_name: lfm-open-license-v1.0",
    "license_link: https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu/blob/main/MODEL_LICENSE",
    "base_model: LiquidAI/LFM2.5-350M",
    "language: [ja, en]",
    "tags:",
    "  - onnx",
    "  - qnn",
    "  - qualcomm",
    "  - npu",
    "  - qcs6490",
    "  - lfm2.5",
    "  - quantized",
]
GITHUB_REPOSITORY_URL = (
    "https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu"
)
HUGGING_FACE_MODEL_URL = (
    "https://huggingface.co/PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_utf8(path: Path, errors: list[str], label: str) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"{label}: not UTF-8: {exc}")
        return None


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    opening_re = re.compile(
        r"^\s*((" + re.escape(TICK) + r"{3,}|~{3,}))(.*)$"
    )
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        opening = opening_re.match(lines[index])
        if not opening:
            index += 1
            continue
        marker = opening.group(1)
        language = opening.group(3).strip()
        body: list[str] = []
        index += 1
        closing = re.compile(
            r"^\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$"
        )
        while index < len(lines) and not closing.match(lines[index]):
            body.append(lines[index])
            index += 1
        if index == len(lines):
            body.append("<UNCLOSED>")
        else:
            index += 1
        blocks.append((language, "\n".join(body)))
    return blocks


def table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return stripped[1:-1].split("|")


def table_shapes(text: str) -> list[tuple[int, ...]]:
    lines = text.splitlines()
    shapes: list[tuple[int, ...]] = []
    index = 0
    while index + 1 < len(lines):
        header = table_cells(lines[index])
        delimiter = table_cells(lines[index + 1])
        if (
            header and len(header) == len(delimiter)
            and all(TABLE_DELIMITER.fullmatch(cell.strip()) for cell in delimiter)
        ):
            rows = [len(header), len(delimiter)]
            index += 2
            while index < len(lines):
                cells = table_cells(lines[index])
                if not cells:
                    break
                rows.append(len(cells))
                index += 1
            shapes.append(tuple(rows))
            continue
        index += 1
    return shapes


def counterpart(path: Path) -> Path:
    if path.name.endswith(".ja.md"):
        return path.with_name(path.name[:-6] + ".md")
    return path.with_name(path.stem + ".ja.md")


def expected_crosslink(path: Path) -> str:
    other = counterpart(path).name
    if path.name.endswith(".ja.md"):
        return f"**English version -> [{other}]({other})**"
    return f"**\u65e5\u672c\u8a9e\u7248 -> [{other}]({other})**"


def crosslink_index(text: str) -> int:
    lines = text.splitlines()
    index = 0
    if lines and lines[0] == "---":
        try:
            index = lines.index("---", 1) + 1
        except ValueError:
            return 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def without_crosslink(text: str) -> str:
    lines = text.splitlines()
    index = crosslink_index(text)
    if index < len(lines):
        del lines[index]
    return "\n".join(lines)


def inline_literals(text: str) -> Counter[str]:
    values = INLINE_CODE.findall(without_crosslink(text))
    return Counter(value.replace(".ja.md", ".md") for value in values)


def audit_doc_pairs(root: Path, errors: list[str], checks: dict[str, int]) -> None:
    markdown = sorted(root.rglob("*.md"))
    pairs: set[tuple[Path, Path]] = set()
    for path in markdown:
        other = counterpart(path)
        if not other.is_file():
            errors.append(f"docs: missing counterpart for {rel(path, root)}")
            continue
        pair = tuple(sorted((path, other)))
        if pair in pairs:
            continue
        pairs.add(pair)
        ja = path if path.name.endswith(".ja.md") else other
        en = other if path.name.endswith(".ja.md") else path
        ja_text = read_utf8(ja, errors, f"docs:{rel(ja, root)}")
        en_text = read_utf8(en, errors, f"docs:{rel(en, root)}")
        if ja_text is None or en_text is None:
            continue
        for doc, text in ((ja, ja_text), (en, en_text)):
            lines = text.splitlines()
            index = crosslink_index(text)
            first = lines[index] if index < len(lines) else ""
            if first != expected_crosslink(doc):
                errors.append(
                    f"docs:{rel(doc, root)}: invalid leading crosslink"
                )
        label = f"docs:{rel(ja, root)} <-> {rel(en, root)}"
        if [len(x) for x in HEADING.findall(ja_text)] != [
            len(x) for x in HEADING.findall(en_text)
        ]:
            errors.append(f"{label}: heading-level sequence differs")
        if table_shapes(ja_text) != table_shapes(en_text):
            errors.append(f"{label}: table shape differs")
        ja_numbers = Counter(NUMBER.findall(ja_text))
        en_numbers = Counter(NUMBER.findall(en_text))
        if ja_numbers != en_numbers:
            errors.append(f"{label}: numeric literals differ: ja={ja_numbers} en={en_numbers}")
        if fenced_blocks(ja_text) != fenced_blocks(en_text):
            errors.append(f"{label}: fenced command/code blocks differ")
        ja_code = inline_literals(ja_text)
        en_code = inline_literals(en_text)
        if ja_code != en_code:
            errors.append(
                f"{label}: inline command/path literals differ: ja={ja_code} en={en_code}"
            )
    checks["markdown_files"] = len(markdown)
    checks["document_pairs"] = len(pairs)


def audit_publication_metadata(
    root: Path, errors: list[str], checks: dict[str, int]
) -> None:
    card_paths = [
        root / "release" / "MODEL_CARD.ja.md",
        root / "release" / "MODEL_CARD.md",
    ]
    readme_paths = [root / "README.ja.md", root / "README.md"]
    for path in card_paths:
        text = read_utf8(path, errors, f"metadata:{rel(path, root)}")
        if text is None:
            continue
        lines = text.splitlines()
        try:
            closing = lines.index("---", 1) if lines and lines[0] == "---" else -1
        except ValueError:
            closing = -1
        if closing < 0 or lines[1:closing] != HF_MODEL_CARD_FRONT_MATTER:
            errors.append(f"metadata:{rel(path, root)}: invalid HF front matter")
        if GITHUB_REPOSITORY_URL not in text:
            errors.append(f"metadata:{rel(path, root)}: missing GitHub link")
    for path in readme_paths:
        text = read_utf8(path, errors, f"metadata:{rel(path, root)}")
        if text is not None and HUGGING_FACE_MODEL_URL not in text:
            errors.append(f"metadata:{rel(path, root)}: missing HF model-card link")

    checklist_expectations = {
        root / "release" / "UPLOAD_CHECKLIST.ja.md": [
            "6. Hugging Face model repository\u3092\u4f5c\u6210\u3059\u308b\u3002",
            "7. model staging\u306e11\u8cc7\u7523\u3092Hugging Face\u3078upload\u3059\u308b\u3002",
            "8. \u516c\u958b\u5148\u306e`asset-manifest.json`\u304b\u308911\u8cc7\u7523\u3092\u518d\u53d6\u5f97\u3057\u3001size\u3068SHA-256\u3092\u518d\u691c\u8a3c\u3059\u308b\u3002",
            "9. `release/MODEL_CARD.md`\u3092Hugging Face\u306e`README.md`\u3068\u3057\u3066upload\u3059\u308b\u3002",
        ],
        root / "release" / "UPLOAD_CHECKLIST.md": [
            "6. Create the Hugging Face model repository.",
            "7. Upload the 11 model-staging assets to Hugging Face.",
            "8. Re-download the 11 assets from the published `asset-manifest.json` and re-verify size and SHA-256.",
            "9. Upload `release/MODEL_CARD.md` as the Hugging Face `README.md`.",
        ],
    }
    for path, expected_lines in checklist_expectations.items():
        text = read_utf8(path, errors, f"metadata:{rel(path, root)}")
        if text is None:
            continue
        positions = [text.find(line) for line in expected_lines]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append(
                f"metadata:{rel(path, root)}: HF upload sequence is incomplete"
            )

    disclosure_paths = [
        root / "README.ja.md",
        root / "README.md",
        root / "release" / "MODEL_CARD.ja.md",
        root / "release" / "MODEL_CARD.md",
        root / "NOTICE.ja.md",
        root / "NOTICE.md",
    ]
    for path in disclosure_paths:
        text = read_utf8(path, errors, f"metadata:{rel(path, root)}")
        if text is None:
            continue
        unofficial = "\u975e\u516c\u5f0f" if path.name.endswith(".ja.md") else "not an official"
        if unofficial not in text:
            errors.append(f"metadata:{rel(path, root)}: missing unofficial notice")
        if "OpenAI Codex" not in text or "Anthropic Claude Code" not in text:
            errors.append(f"metadata:{rel(path, root)}: missing AI-use disclosure")
    checks["hf_model_cards"] = len(card_paths)
    checks["publication_crosslinks"] = len(card_paths) + len(readme_paths)
    checks["hf_upload_checklists"] = len(checklist_expectations)
    checks["unofficial_disclosures"] = len(disclosure_paths)
    checks["ai_assistance_disclosures"] = len(disclosure_paths)


def audit_links(root: Path, errors: list[str], checks: dict[str, int]) -> None:
    checked = 0
    for path in sorted(root.rglob("*.md")):
        text = read_utf8(path, errors, f"links:{rel(path, root)}")
        if text is None:
            continue
        for raw in LINK.findall(text):
            target = raw.strip().strip("<>")
            if not target or target.startswith("#") or target.startswith(EXTERNAL_SCHEMES):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = (path.parent / unquote(target)).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"links:{rel(path, root)}: target escapes repository: {raw}")
                continue
            if not resolved.exists():
                errors.append(f"links:{rel(path, root)}: missing target: {raw}")
    checks["internal_links"] = checked


def private_patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("Windows user path", re.compile(
            r"[A-Za-z]:[\\/]+Users[\\/]+", re.I
        )),
        ("device home path", re.compile(
            r"/home/(?:" + "rad" + r"xa|root)(?:/|\b)", re.I
        )),
        ("device login reference", re.compile(r"\b" + "rad" + "xa" + r"@", re.I)),
        ("private run path", re.compile("/mnt/" + "q6a_sd", re.I)),
        ("private IPv4 address", re.compile(
            r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
            r"192\.168\.(?:\d{1,3}\.)\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b"
        )),
        ("private key material", re.compile("BEGIN " + "PRIVATE KEY", re.I)),
        ("password helper", re.compile(r"\b" + "ssh" + "pass" + r"\b", re.I)),
        ("credential assignment", re.compile(
            r"(?im)^\s*(?:password|passwd|api[_-]?key|secret|token)"
            r"\s*[:=]\s*[\"']?[^\s\"'<>]{4,}"
        )),
        ("email address", re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
        )),
    ]


def audit_repository(root: Path, errors: list[str], checks: dict[str, int]) -> None:
    files = visible_files(root)
    checks["repository_files"] = len(files)
    checks["repository_bytes"] = sum(path.stat().st_size for path in files)
    for path in files:
        relative_path = rel(path, root)
        bad_dirs = sorted(set(path.relative_to(root).parts) & FORBIDDEN_DIRS)
        if bad_dirs:
            errors.append(f"repository:{relative_path}: generated directory {bad_dirs}")
        suffix = path.suffix.lower()
        if suffix in REPO_BINARY_SUFFIXES:
            errors.append(f"repository:{relative_path}: forbidden binary suffix {suffix}")
        if path.stat().st_size > MAX_REPO_BYTES:
            errors.append(f"repository:{relative_path}: exceeds {MAX_REPO_BYTES} bytes")
        if suffix not in TEXT_SUFFIXES:
            continue
        text = read_utf8(path, errors, f"repository:{relative_path}")
        if text is None:
            continue
        for name, pattern in private_patterns():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"repository:{relative_path}:{line}: {name}")


def load_json(path: Path, errors: list[str], label: str) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return None


def audit_assets(
    root: Path, assets: Path, errors: list[str], checks: dict[str, int]
) -> None:
    repo_manifest = load_json(
        root / "runner" / "config" / "model-assets.json", errors, "assets:repo manifest"
    )
    staging_manifest = load_json(
        assets / "asset-manifest.json", errors, "assets:staging manifest"
    )
    if not isinstance(repo_manifest, dict) or not isinstance(staging_manifest, dict):
        return
    if repo_manifest != staging_manifest:
        errors.append("assets: repository and staging manifests differ")
    entries = repo_manifest.get("files")
    if not isinstance(entries, list):
        errors.append("assets: manifest files must be a list")
        return
    checks["manifest_entries"] = len(entries)
    if len(entries) != 11:
        errors.append(f"assets: expected 11 manifest entries, got {len(entries)}")
    expected = {"asset-manifest.json"}
    for index, item in enumerate(entries):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append(f"assets: invalid entry {index}")
            continue
        value = item["path"]
        expected.add(Path(value).as_posix())
        path = assets / Path(value)
        try:
            path.resolve().relative_to(assets.resolve())
        except ValueError:
            errors.append(f"assets: entry escapes staging: {value}")
            continue
        if not path.is_file():
            errors.append(f"assets: missing file: {value}")
            continue
        if path.stat().st_size != item.get("size"):
            errors.append(f"assets:{value}: size differs from manifest")
        if sha256_file(path) != item.get("sha256"):
            errors.append(f"assets:{value}: SHA-256 differs from manifest")
    actual = {rel(path, assets) for path in visible_files(assets)}
    if actual != expected:
        errors.append(
            f"assets: staging set differs: missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}"
        )
    for path in visible_files(assets):
        name = path.name.lower()
        if path.suffix.lower() in MODEL_FORBIDDEN_SUFFIXES:
            errors.append(f"assets:{rel(path, assets)}: forbidden distributable suffix")
        if "epcontext" in name or (name.startswith("qnn") and path.suffix.lower() == ".bin"):
            errors.append(f"assets:{rel(path, assets)}: generated QNN context is forbidden")
    checks["staging_files"] = len(actual)
    checks["staging_bytes"] = sum(path.stat().st_size for path in visible_files(assets))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    errors: list[str] = []
    checks: dict[str, int] = {}
    if not root.is_dir():
        errors.append(f"repository root does not exist: {root}")
    else:
        audit_repository(root, errors, checks)
        audit_doc_pairs(root, errors, checks)
        audit_links(root, errors, checks)
        audit_publication_metadata(root, errors, checks)
    if args.assets_dir is not None:
        assets = args.assets_dir.resolve()
        if not assets.is_dir():
            errors.append(f"assets directory does not exist: {assets}")
        else:
            audit_assets(root, assets, errors, checks)
    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
        "notes": {
            "publication_namespace": "PrismPhi",
            "external_urls": "Recorded but not fetched by this offline audit.",
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
