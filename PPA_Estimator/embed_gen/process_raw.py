#!/usr/bin/env python3
"""Generate enriched layout JSON with routing renaming and supply rail detection."""
import copy
import json
import os
import re
import sys
from typing import Dict, List


def parse_net_file(net_file: str) -> Dict[str, int]:
    """Parse a SPICE-like .net file and map FET names to their nfin values."""
    pattern = r"(MM\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+w=\S+\s+l=\S+\s+nfin=(\d+)"
    nfin_dict: Dict[str, int] = {}

    with open(net_file, "r", encoding="utf-8") as f:
        content = f.read()

    for fet_name, nfin in re.findall(pattern, content):
        nfin_dict[fet_name] = int(nfin)

    return nfin_dict


def modify_routing_bounding_boxes(routing_data: List[Dict]) -> List[Dict]:
    """Apply layer-specific coordinate transforms to routing boxes, preserving net names."""
    modified_data = copy.deepcopy(routing_data)

    for item in modified_data:
        layer = item["layer"]
        box = item["box"]

        if layer == 0:
            box["x1"] = box["x1"] * 13.5 - 9
            box["x2"] = box["x2"] * 13.5 + 9
            box["y1"] = box["y1"] * 9 - 9
            box["y2"] = box["y2"] * 9 + 9
        elif layer == 1:
            box["y1"] = box["y1"] * 9 - 9
            box["y2"] = box["y2"] * 9 + 9
            box["x1"] = box["x1"] * 13.5
            box["x2"] = box["x2"] * 13.5
        elif layer == 2:
            box["x1"] = box["x1"] * 13.5 - 9
            box["x2"] = box["x2"] * 13.5 + 9
            box["y1"] = box["y1"] * 9 - 9
            box["y2"] = box["y2"] * 9 + 9
        elif layer == 3:
            box["x1"] = box["x1"] * 13.5 - 9
            box["x2"] = box["x2"] * 13.5 + 9
            box["y1"] = box["y1"] * 9 - 5   
            box["y2"] = box["y2"] * 9 + 5
        elif layer == 4:
            box["x1"] = box["x1"] * 13.5 - 9
            box["x2"] = box["x2"] * 13.5 + 9
            box["y1"] = box["y1"] * 9 - 9
            box["y2"] = box["y2"] * 9 + 9
        elif layer == 5:
            box["y1"] = box["y1"] * 9 - 9
            box["y2"] = box["y2"] * 9 + 9
            box["x1"] = box["x1"] * 13.5
            box["x2"] = box["x2"] * 13.5

    return modified_data


def generate_fet_bounding_boxes(fets_data: List[Dict], nfin_dict: Dict[str, int]) -> List[Dict]:
    """Create bounding boxes for each FET terminal using ASAP7 placement rules."""
    bounding_boxes: List[Dict] = []

    for fet in fets_data:
        fet_name = fet["fet_name"]
        x = fet["location"]["x"]
        flip = fet["location"]["flip"]
        np_flag = fet["np"]
        nfin = nfin_dict.get(fet_name, fet.get("nfin", 1))
        nets = fet["nets"]

        x_base = 3 * 13.5 + (x / 2) * 54
        y1_min = 5 * 9 - 9
        y2_max = 21 * 9 + 9

        if np_flag:
            if nfin == 2:
                y1 = 21 * 9 - 45
                y2 = 21 * 9 + 9
            else:
                y1 = 21 * 9 - 18
                y2 = 21 * 9 + 9
        else:
            if nfin == 2:
                y1 = 5 * 9 - 9
                y2 = 5 * 9 + 45
            else:
                y1 = 5 * 9 - 9
                y2 = 5 * 9 + 18

        if not flip:
            d_box = {"x1": x_base - 35, "x2": x_base - 10, "y1": y1, "y2": y2}
            g_box = {"x1": x_base - 10, "x2": x_base + 10, "y1": y1_min, "y2": y2_max}
            s_box = {"x1": x_base + 10, "x2": x_base + 35, "y1": y1, "y2": y2}
        else:
            d_box = {"x1": x_base + 10, "x2": x_base + 35, "y1": y1, "y2": y2}
            g_box = {"x1": x_base - 10, "x2": x_base + 10, "y1": y1_min, "y2": y2_max}
            s_box = {"x1": x_base - 35, "x2": x_base - 10, "y1": y1, "y2": y2}

        bounding_boxes.extend([
            {"fet_name": fet_name, "terminal": "D", "net_name": nets["D"], "box": d_box},
            {"fet_name": fet_name, "terminal": "G", "net_name": nets["G"], "box": g_box},
            {"fet_name": fet_name, "terminal": "S", "net_name": nets["S"], "box": s_box},
        ])

    return bounding_boxes


def rename_routing_boxes(routing_boxes: List[Dict]) -> Dict[str, Dict]:
    """Rename routing boxes following routing_{layer}_{number} convention."""
    renamed: Dict[str, Dict] = {}
    per_layer_counter: Dict[int, int] = {}

    for item in routing_boxes:
        layer = item["layer"]
        per_layer_counter[layer] = per_layer_counter.get(layer, 0) + 1
        name = f"routing_{layer}_{per_layer_counter[layer]}"
        renamed[name] = {
            "layer": layer,
            "box": copy.deepcopy(item["box"]),
        }
        if "net_name" in item:
            renamed[name]["net_name"] = item["net_name"]

    return renamed


def add_routing_coupling_features(routing_named: Dict[str, Dict]) -> None:
    """Augment routing entries with lateral and vertical coupling metrics."""
    centroids: Dict[str, Dict[str, float]] = {}
    layer_map: Dict[int, List[str]] = {}

    for name, data in routing_named.items():
        box = data.get("box")
        if not box:
            continue
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        centroids[name] = {"x": cx, "y": cy}
        data["lat_cpl"] = 0.0
        data["ver_cpl"] = 0.0

        layer = data.get("layer")
        if layer is None:
            continue
        layer_map.setdefault(layer, []).append(name)

    # Lateral coupling within the same layer based on overlapping projections.
    for layer, names in layer_map.items():
        count = len(names)
        for i in range(count):
            name_i = names[i]
            data_i = routing_named[name_i]
            box_i = data_i["box"]
            centroid_i = centroids[name_i]
            for j in range(i + 1, count):
                name_j = names[j]
                data_j = routing_named[name_j]
                box_j = data_j["box"]
                centroid_j = centroids[name_j]

                overlap_x = min(box_i["x2"], box_j["x2"]) - max(box_i["x1"], box_j["x1"])
                if overlap_x > 0:
                    width = abs(centroid_i["y"] - centroid_j["y"])
                    if width > 0:
                        coupling = overlap_x / width
                        data_i["lat_cpl"] += coupling
                        data_j["lat_cpl"] += coupling

                overlap_y = min(box_i["y2"], box_j["y2"]) - max(box_i["y1"], box_j["y1"])
                if overlap_y > 0:
                    width = abs(centroid_i["x"] - centroid_j["x"])
                    if width > 0:
                        coupling = overlap_y / width
                        data_i["lat_cpl"] += coupling
                        data_j["lat_cpl"] += coupling

    # Vertical coupling across layers when areas overlap.
    sorted_layers = sorted(layer_map.keys())
    for idx, layer in enumerate(sorted_layers):
        names_layer = layer_map[layer]
        for other_layer in sorted_layers[idx + 1 :]:
            gap = abs(other_layer - layer)
            if gap == 0:
                continue
            names_other = layer_map[other_layer]
            for name_i in names_layer:
                data_i = routing_named[name_i]
                box_i = data_i["box"]
                for name_j in names_other:
                    data_j = routing_named[name_j]
                    box_j = data_j["box"]

                    overlap_x = min(box_i["x2"], box_j["x2"]) - max(box_i["x1"], box_j["x1"])
                    overlap_y = min(box_i["y2"], box_j["y2"]) - max(box_i["y1"], box_j["y1"])
                    if overlap_x <= 0 or overlap_y <= 0:
                        continue

                    area_overlap = overlap_x * overlap_y
                    coupling = area_overlap / gap
                    data_i["ver_cpl"] += coupling
                    data_j["ver_cpl"] += coupling


def restructure_fets(fets_data: List[Dict], fet_bounding_boxes: List[Dict]) -> Dict[str, Dict]:
    """Group FET information keyed by original fet_name with terminal details."""
    structured: Dict[str, Dict] = {}

    for fet in fets_data:
        fet_name = fet["fet_name"]
        structured[fet_name] = {
            "fet_name": fet_name,
            "location": copy.deepcopy(fet["location"]),
            "np": fet["np"],
            "nfin": fet.get("nfin", 1),
            "terminals": {},
        }

    for terminal_box in fet_bounding_boxes:
        fet_name = terminal_box["fet_name"]
        terminal = terminal_box["terminal"]
        structured[fet_name]["terminals"][terminal] = {
            "net": terminal_box["net_name"],
            "box": copy.deepcopy(terminal_box["box"]),
        }

    return structured


def extract_supply_boxes(layout_data: List[Dict]) -> Dict[str, Dict]:
    """Select supply boxes from layers 16 and 17 only, and tag VDD/VSS rails on layer 16."""
    supplies: Dict[str, Dict] = {}
    special: Dict[str, str] = {}
    seen_coords: set = set()
    counter = 1

    def add_supply(item: Dict, tag: str = None) -> str:
        nonlocal counter
        box = copy.deepcopy(item["box"])
        coords = (item.get("layer"), box["x1"], box["x2"], box["y1"], box["y2"])
        if coords in seen_coords:
            if tag:
                for name, data in supplies.items():
                    if data["_coords"] == coords:
                        data["tag"] = tag
                        special[tag] = name
                        break
            return ""

        seen_coords.add(coords)
        name = f"supply_{counter}"
        counter += 1
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        supplies[name] = {
            "layer": item.get("layer"),
            "datatype": item.get("datatype"),
            "box": box,
            "width": width,
            "height": height,
            "_coords": coords,
        }
        if tag:
            supplies[name]["tag"] = tag
            special[tag] = name
        return name

    # Only process layers 16 and 17 for supply rails
    for item in layout_data:
        layer = item.get("layer")
        if layer not in [16, 17]:
            continue
            
        box = item.get("box")
        if not box:
            continue
        height = box["y2"] - box["y1"]
        if height > 25:
            tag = None
            if layer == 16 and (box["x2"] - box["x1"]) > 30:
                if box["y1"] == 208 and box["y2"] == 224:
                    tag = "VDD"
                elif box["y1"] == -8 and box["y2"] == 8:
                    tag = "VSS"
            add_supply(item, tag)

    # Second pass for VDD/VSS identification, still only on layer 16
    if "VDD" not in special or "VSS" not in special:
        for item in layout_data:
            if item.get("layer") != 16:
                continue
            box = item.get("box")
            if not box:
                continue
            width = box["x2"] - box["x1"]
            if width <= 30:
                continue
            if box["y1"] == 208 and box["y2"] == 224 and "VDD" not in special:
                add_supply(item, "VDD")
            elif box["y1"] == -8 and box["y2"] == 8 and "VSS" not in special:
                add_supply(item, "VSS")
            if "VDD" in special and "VSS" in special:
                break

    for data in supplies.values():
        data.pop("_coords", None)

    return {"supplies": supplies, "identified": special}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_single_layout(net_file: str, fet_file: str, layout_file: str, routing_file: str) -> Dict:
    routing_data = load_json(routing_file)
    fets_data = load_json(fet_file)
    layout_data = load_json(layout_file)

    nfin_dict = parse_net_file(net_file)
    for fet in fets_data:
        fet_name = fet["fet_name"]
        if fet_name in nfin_dict:
            fet["nfin"] = nfin_dict[fet_name]

    modified_routing = modify_routing_bounding_boxes(routing_data)
    routing_named = rename_routing_boxes(modified_routing)
    add_routing_coupling_features(routing_named)

    fet_bboxes = generate_fet_bounding_boxes(fets_data, nfin_dict)
    fets_structured = restructure_fets(fets_data, fet_bboxes)

    supply_info = extract_supply_boxes(layout_data)

    metadata = {
        "description": "Complete layout aggregation with routing renaming and supply rail tagging",
        "source_files": {
            "net": os.path.basename(net_file),
            "fets": os.path.basename(fet_file),
            "layout": os.path.basename(layout_file),
            "routing": os.path.basename(routing_file),
        },
        "counts": {
            "routing_boxes": len(routing_named),
            "fets": len(fets_structured),
            "fet_terminals": sum(len(fet["terminals"]) for fet in fets_structured.values()),
            "supplies": len(supply_info["supplies"]),
            "supply_layers": "16, 17 only",
        },
    }

    return {
        "metadata": metadata,
        "routing": routing_named,
        "fets": fets_structured,
        "layout": supply_info,
    }


def collect_dir_entries(dir_path: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for name in os.listdir(dir_path):
        if not name.endswith(".json"):
            continue
        match = re.search(r"_(\d+)\.json$", name)
        if not match:
            continue
        idx = match.group(1)
        full_path = os.path.join(dir_path, name)
        entries[idx] = full_path
    return entries


def main(argv: List[str]) -> int:
    if len(argv) < 5:
        print("Usage: python process_raw.py <netlist.net> <fets_path/fets_dir> <layout_path/layout_dir> <routing_path/routing_dir> [output_path/output_dir]")
        print("i.e. python process_raw.py ./AOI22xp25.net ./fet/AOI22xp25_ASAP7_6t_L_fets_0.json ./layout/AOI22xp25_ASAP7_6t_L_layout_0.json ./routing/AOI22xp25_ASAP7_6t_L_routing_0.json ./processed_raw.json")
        
        return 1

    net_path, fet_path, layout_path, routing_path = argv[1:5]
    output_path = argv[5] if len(argv) > 5 else None

    if not os.path.isfile(net_path):
        print(f"Netlist not found: {net_path}")
        return 1

    paths = [fet_path, layout_path, routing_path]
    types = [os.path.isdir(p) for p in paths]

    if all(not is_dir for is_dir in types):
        if output_path is None:
            output_path = "processed_raw.json"
        result = process_single_layout(net_path, fet_path, layout_path, routing_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"Wrote combined layout to {output_path}")
        return 0

    if all(types):
        if output_path is None:
            output_path = "processed_raw_outputs" 
        os.makedirs(output_path, exist_ok=True)

        fet_entries = collect_dir_entries(fet_path)
        layout_entries = collect_dir_entries(layout_path)
        routing_entries = collect_dir_entries(routing_path)

        common_keys = sorted(set(fet_entries) & set(layout_entries) & set(routing_entries), key=lambda x: int(x))
        if not common_keys:
            print("No matching JSON files across provided directories.")
            return 1

        for key in common_keys:
            fet_file = fet_entries[key]
            layout_file = layout_entries[key]
            routing_file = routing_entries[key]
            result = process_single_layout(net_path, fet_file, layout_file, routing_file)
            out_file = os.path.join(output_path, f"complete_layout_{key}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            print(f"Wrote {out_file}")
        return 0

    print("Mixed file and directory inputs detected; provide either all files or all directories for fets/layout/routing.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
