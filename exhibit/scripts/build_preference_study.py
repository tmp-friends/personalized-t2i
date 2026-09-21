#!/usr/bin/env python3
"""Build the blind preference study: collection page, participants, manifest, pages.

Without ``participants.json`` this writes only the collection page and the
Japanese instructions and records ``comparison_images: not_generated``. It never
invents a participant, an answer, or a win rate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.config import ASSETS
from exhibit.evaluation import load_evaluation_config, study_manifest_hash

from exhibit import study as study_lib


def load_config(args):
    config = load_evaluation_config(args.config, "study")
    study = config["study"]
    if args.study_dir:
        study["study_dir"] = str(Path(args.study_dir).resolve())
    if args.study_id:
        study["study_id"] = args.study_id
    return config


def write_collection(config, catalogs, directory, *, assets=ASSETS):
    page = study_lib.build_collection_page(config, catalogs)
    study_lib.write_text(directory / "collect/index.html", page["html"])
    for relative, source in page["images"].items():
        target = directory / "collect" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(assets) / source, target)
    return len(page["images"])


def load_images(directory, manifest):
    path = directory / "images.json"
    if not path.is_file() or manifest is None:
        return {}
    value = json.loads(path.read_text())
    if value.get("study_hash") != manifest["study_hash"]:
        return {}
    return value.get("images", {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-dir", help="override the configured study directory")
    parser.add_argument("--study-id")
    parser.add_argument(
        "--merge",
        nargs="+",
        default=[],
        metavar="EXPORT",
        help="validate participant exports and merge them into participants.json",
    )
    args = parser.parse_args(argv)

    config = load_config(args)
    study = config["study"]
    directory = Path(study["study_dir"])
    catalogs = study_lib.catalogs_for(config)
    report = {"study_id": study["study_id"], "study_kind": study["study_kind"]}

    directory.mkdir(parents=True, exist_ok=True)
    report["collection_images"] = write_collection(config, catalogs, directory)

    participants_path = directory / "participants.json"
    if args.merge:
        exports = []
        for name in args.merge:
            path = Path(name)
            try:
                exports.append((path.name, json.loads(path.read_text())))
            except (OSError, ValueError) as error:
                exports.append((path.name, {"__unreadable__": str(error)}))
        existing = (
            json.loads(participants_path.read_text())
            if participants_path.is_file()
            else None
        )
        document, merge_report = study_lib.merge_participants(
            config, exports, existing=existing, catalogs=catalogs
        )
        study_lib.write_json(participants_path, document)
        study_lib.write_json(directory / "merge-report.json", merge_report)
        report["merge"] = merge_report
        print(json.dumps({"type": "merge", **merge_report}, ensure_ascii=False))

    participants = (
        json.loads(participants_path.read_text())
        if participants_path.is_file()
        else None
    )
    manifest = None
    failure = None
    if participants is not None:
        try:
            manifest, keys = study_lib.build_manifest(
                config, participants, catalogs=catalogs
            )
        except study_lib.StudyError as error:
            failure = str(error)
        else:
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text())
                if (
                    previous.get("study_hash") != manifest["study_hash"]
                    and (directory / "images.json").is_file()
                ):
                    failure = (
                        "a generated study manifest already exists with another hash; "
                        "use a separate --study-dir instead of rewriting it"
                    )
                    manifest = None
                elif previous.get("study_hash") == manifest["study_hash"]:
                    manifest = previous
            if manifest is not None:
                if study_manifest_hash(manifest) != manifest["study_hash"]:
                    raise SystemExit("study manifest hash mismatch")
                study_lib.write_json(manifest_path, manifest)
                study_lib.write_json(directory / "keys.json", keys)
                report["image_count"] = manifest["image_count"]
                report["participants"] = len(manifest["participants"])

    pages = 0
    if manifest is not None:
        images = load_images(directory, manifest)
        if images:
            keys = json.loads((directory / "keys.json").read_text())
            built = study_lib.build_answer_pages(
                config, manifest, keys, images, directory
            )
            pages = len(built["pages"])
            report["answer_pages"] = built
        else:
            report["answer_pages"] = "not_generated"

    status = study_lib.status_document(
        config,
        participants=participants,
        manifest=manifest,
        images=len(load_images(directory, manifest)) if manifest else 0,
        pages=pages,
    )
    study_lib.write_json(directory / "status.json", status)
    study_lib.write_text(
        directory / "README.md", study_lib.instructions_markdown(config, status=status)
    )
    report["status"] = status
    print(json.dumps({"type": "study_build", **report}, ensure_ascii=False))
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
