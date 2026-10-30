"""python -m pii_pipeline.cli --config configs/pipeline.yaml (--text "..." | --file path) [--json] [--mapping]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import PIIPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pii-pipeline")
    parser.add_argument("--config", required=True, type=Path)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file", type=Path)
    parser.add_argument("--language", default=None)
    parser.add_argument("--json", action="store_true", help="print masked text, entities and (if --mapping) the mapping as JSON")
    parser.add_argument("--mapping", action="store_true", help="include placeholder -> original mapping (contains PII)")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    result = PIIPipeline.from_yaml(args.config).mask(text, language=args.language, return_mapping=args.mapping)

    sys.stdout.reconfigure(encoding="utf-8")
    if args.json:
        payload = {"text": result.text, "entities": result.entities, "rejected": result.rejected}
        if args.mapping:
            payload["mapping"] = result.mapping
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
