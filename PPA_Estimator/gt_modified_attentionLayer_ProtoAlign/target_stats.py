import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

METRIC_KEYS = (
    "leakage",
    "internal_power",
    "cell_rise",
    "cell_fall",
    "rise_transition",
    "fall_transition",
)


def _parse_values_string(values_str: str) -> List[float]:
    return [float(part.strip()) for part in values_str.split(",") if part.strip()]


def extract_metrics(data: Dict) -> Dict[str, Optional[Union[List[float], float]]]:
    leakage = None
    if "leakage_power" in data and "cell_leakage_power" in data["leakage_power"]:
        leakage = float(data["leakage_power"]["cell_leakage_power"])

    internal_power = None
    internal_section = data.get("internal_power")
    if internal_section and internal_section.get("values"):
        internal_power = _parse_values_string(internal_section["values"][0])

    timing = data.get("timing", {})
    cell_rise = None
    cell_fall = None
    rise_transition = None
    fall_transition = None

    if timing:
        if timing.get("cell_rise", {}).get("values"):
            cell_rise = _parse_values_string(timing["cell_rise"]["values"][0])
        if timing.get("cell_fall", {}).get("values"):
            cell_fall = _parse_values_string(timing["cell_fall"]["values"][0])
        if timing.get("rise_transition", {}).get("values"):
            rise_transition = _parse_values_string(timing["rise_transition"]["values"][0])
        if timing.get("fall_transition", {}).get("values"):
            fall_transition = _parse_values_string(timing["fall_transition"]["values"][0])

    return {
        "leakage": leakage,
        "internal_power": internal_power,
        "cell_rise": cell_rise,
        "cell_fall": cell_fall,
        "rise_transition": rise_transition,
        "fall_transition": fall_transition,
    }


def _init_metric(value):
    if isinstance(value, list):
        return [0.0] * len(value), 0
    return 0.0, 0


def _accumulate(existing_sum, count, value):
    if isinstance(value, list):
        if not value:
            return existing_sum, count
        if not existing_sum:
            existing_sum = [0.0] * len(value)
        for idx, val in enumerate(value):
            existing_sum[idx] += val
    else:
        existing_sum += value
    count += 1
    return existing_sum, count


def _compute_mean(sum_value, count):
    if count == 0:
        return [] if isinstance(sum_value, list) else float("nan")
    if isinstance(sum_value, list):
        return [val / count for val in sum_value]
    return sum_value / count


def format_metric(value):
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-cell and overall metrics from target JSONs")
    parser.add_argument("target_dir", type=Path, help="Directory containing target JSON files")
    parser.add_argument("--output", type=Path, default=None, help="Path to write aggregated statistics as JSON")
    parser.add_argument("--per-cell", action="store_true", help="Include per-cell averages in the output")
    parser.add_argument(
        "--log1p-leakage",
        action="store_true",
        help="Apply log1p transform to leakage before computing statistics",
    )
    parser.add_argument(
        "--log1p-internal",
        action="store_true",
        help="Apply log1p transform to internal power before computing statistics",
    )
    args = parser.parse_args()

    target_dir = args.target_dir
    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")

    def _init_accumulator(value: Union[List[float], float]) -> Dict[str, Union[List[float], float, int]]:
        if isinstance(value, list):
            zeros = [0.0] * len(value)
            return {"sum": zeros[:], "sumsq": zeros[:], "count": 0}
        return {"sum": 0.0, "sumsq": 0.0, "count": 0}

    def _add_value(acc: Dict[str, Union[List[float], float, int]], value: Union[List[float], float]) -> None:
        if isinstance(value, list):
            if not value:
                return
            if not acc["sum"]:
                acc["sum"] = [0.0] * len(value)
                acc["sumsq"] = [0.0] * len(value)
            for idx, val in enumerate(value):
                acc["sum"][idx] += val
                acc["sumsq"][idx] += val * val
        else:
            acc["sum"] += value
            acc["sumsq"] += value * value
        acc["count"] += 1

    def _finalize(acc: Dict[str, Union[List[float], float, int]]):
        count = acc["count"]
        if count == 0:
            return None
        sum_val = acc["sum"]
        sumsq_val = acc["sumsq"]
        if isinstance(sum_val, list):
            mean = [val / count for val in sum_val]
            std = [math.sqrt(max(sumsq_val[idx] / count - mean[idx] ** 2, 0.0)) for idx in range(len(mean))]
        else:
            mean = sum_val / count
            std = math.sqrt(max(sumsq_val / count - mean * mean, 0.0))
        return mean, std, count

    per_cell: Dict[str, Dict[str, Dict[str, Union[List[float], float, int]]]] = {}
    overall_accumulators: Dict[str, Dict[str, Union[List[float], float, int]]] = {}

    json_paths = sorted(target_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No JSON files found under {target_dir}")

    for path in json_paths:
        with path.open() as handle:
            data = json.load(handle)
        cell_name = data.get("cell_name", path.stem)
        metrics = extract_metrics(data)

        cell_stats = per_cell.setdefault(cell_name, {})
        for key in METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue

            apply_log1p = (
                (key == "leakage" and args.log1p_leakage)
                or (key == "internal_power" and args.log1p_internal)
            )
            if value is not None and apply_log1p:
                if isinstance(value, list):
                    value = [math.log1p(v) for v in value]
                else:
                    value = math.log1p(value)

            # per-cell accumulation
            if key not in cell_stats:
                cell_stats[key] = _init_accumulator(value)
            _add_value(cell_stats[key], value)

            # overall accumulation
            if key not in overall_accumulators:
                overall_accumulators[key] = _init_accumulator(value)
            _add_value(overall_accumulators[key], value)

    per_cell_output: Dict[str, Dict[str, Dict[str, Union[List[float], float, int]]]] = {}
    if args.per_cell:
        for cell_name in sorted(per_cell.keys()):
            stats = per_cell[cell_name]
            cell_result: Dict[str, List[float] | float] = {}
            for key in METRIC_KEYS:
                if key not in stats or stats[key]["count"] == 0:
                    continue
                finalized = _finalize(stats[key])
                if finalized is None:
                    continue
                mean, std, count = finalized
                cell_result[key] = {"mean": mean, "std": std, "count": count}
            if cell_result:
                per_cell_output[cell_name] = cell_result

    overall_output: Dict[str, Dict[str, Union[List[float], float, int]]] = {}
    for key in METRIC_KEYS:
        accumulator = overall_accumulators.get(key)
        if not accumulator or accumulator["count"] == 0:
            continue
        finalized = _finalize(accumulator)
        if finalized is None:
            continue
        mean, std, count = finalized
        entry: Dict[str, Union[List[float], float, int, str]] = {
            "mean": mean,
            "std": std,
            "count": count,
        }
        if (key == "leakage" and args.log1p_leakage) or (
            key == "internal_power" and args.log1p_internal
        ):
            entry["transform"] = "log1p"
        overall_output[key] = entry

    result = {"overall": overall_output}
    if args.per_cell:
        result["per_cell"] = per_cell_output

    if args.log1p_leakage and "leakage" not in overall_output:
        print("Warning: log1p requested but leakage values were missing in the dataset")
    if args.log1p_internal and "internal_power" not in overall_output:
        print("Warning: log1p requested but internal power values were missing in the dataset")

    if args.output:
        with args.output.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote statistics to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
