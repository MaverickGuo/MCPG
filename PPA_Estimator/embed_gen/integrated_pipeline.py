#!/usr/bin/env python3
"""Run process_raw.py, path_extractor.py, and generate_graph_embeddings_pe.py sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def _run_stage(name: str, command: List[str], workdir: Path) -> float:
    """Run a command as a subprocess and report its duration."""
    print(f"\n=== Running {name} ===")
    print("Command:", " ".join(command))

    start = time.perf_counter()
    try:
        subprocess.run(command, check=True, cwd=workdir)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{name} failed with exit code {exc.returncode}") from exc

    duration = time.perf_counter() - start
    print(f"=== Completed {name} in {duration:.2f} seconds ===")
    return duration


def _ensure_output_dirs(paths: List[Path]) -> None:
    for path in paths:
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Integrate the raw processing, path extraction, and embedding steps"
    )
    parser.add_argument("--net", required=True, help="Path to the .net file")
    parser.add_argument("--fets", required=True, help="Path to the FET JSON or directory")
    parser.add_argument("--layout", required=True, help="Path to the layout JSON or directory")
    parser.add_argument("--routing", required=True, help="Path to the routing JSON or directory")
    parser.add_argument(
        "--processed-output",
        required=True,
        help="Output path for the enriched layout JSON from process_raw.py",
    )
    parser.add_argument(
        "--topology-output",
        required=True,
        help="Output path for the topology JSON from path_extractor.py",
    )
    parser.add_argument(
        "--embeddings-output",
        required=True,
        help="Output path for the embeddings JSON from generate_graph_embeddings_pe.py",
    )
    parser.add_argument(
        "--process-raw-script",
        default=str(Path(__file__).resolve().parent / "process_raw.py"),
        help="Location of process_raw.py (defaults to the copy next to this script)",
    )
    parser.add_argument(
        "--path-extractor-script",
        default=str(Path(__file__).resolve().parent / "path_extractor.py"),
        help="Location of path_extractor.py",
    )
    parser.add_argument(
        "--embedding-script",
        default=str(Path(__file__).resolve().parent / "generate_graph_embeddings_pe.py"),
        help="Location of generate_graph_embeddings_pe.py",
    )
    parser.add_argument(
        "--report-timing",
        action="store_true",
        help="Print a per-stage timing summary after all steps complete",
    )

    args = parser.parse_args(argv)

    process_raw_script = Path(args.process_raw_script).resolve()
    path_extractor_script = Path(args.path_extractor_script).resolve()
    embedding_script = Path(args.embedding_script).resolve()

    workdir = process_raw_script.parent

    net_path = str(Path(args.net).resolve())
    fets_path = str(Path(args.fets).resolve())
    layout_path = str(Path(args.layout).resolve())
    routing_path = str(Path(args.routing).resolve())

    processed_output = Path(args.processed_output).resolve()
    topology_output = Path(args.topology_output).resolve()
    embeddings_output = Path(args.embeddings_output).resolve()
    _ensure_output_dirs([processed_output, topology_output, embeddings_output])

    process_raw_command = [
        sys.executable,
        str(process_raw_script),
        net_path,
        fets_path,
        layout_path,
        routing_path,
        str(processed_output),
    ]

    path_extractor_command = [
        sys.executable,
        str(path_extractor_script),
        str(processed_output),
        str(topology_output),
    ]

    embedding_command = [
        sys.executable,
        str(embedding_script),
        "--net",
        net_path,
        "--layout",
        str(processed_output),
        "--topology",
        str(topology_output),
        "--output",
        str(embeddings_output),
    ]

    durations: Dict[str, float] = {}
    durations["process_raw.py"] = _run_stage("process_raw.py", process_raw_command, workdir)
    durations["path_extractor.py"] = _run_stage(
        "path_extractor.py", path_extractor_command, workdir
    )
    durations["generate_graph_embeddings_pe.py"] = _run_stage(
        "generate_graph_embeddings_pe.py", embedding_command, workdir
    )

    if args.report_timing:
        total = sum(durations.values())
        print("\n=== Timing Summary ===")
        for stage, seconds in durations.items():
            print(f"{stage}: {seconds:.2f} seconds")
        print(f"Total: {total:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
