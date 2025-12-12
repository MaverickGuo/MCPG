#!/usr/bin/env python3
"""
Path Extractor - Enhanced Version with Flexible Routing

Handles complex routing patterns including:
- Same-layer edge connections
- Non-sequential layer traversal (e.g., 5→4→3→2→3→2→1→0)
- Circular path detection
- Bidirectional traversal (downward and upward)
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional


def boxes_overlap(box1: Dict[str, float], box2: Dict[str, float], tolerance: float = 1e-6) -> bool:
    """Check if two bounding boxes overlap with optional tolerance"""
    return not (box1['x2'] <= box2['x1'] + tolerance or 
                box2['x2'] <= box1['x1'] + tolerance or 
                box1['y2'] <= box2['y1'] + tolerance or 
                box2['y2'] <= box1['y1'] + tolerance)


def boxes_share_edge(box1: Dict[str, float], box2: Dict[str, float], tolerance: float = 1e-6) -> bool:
    """Check if two boxes share an edge (are adjacent but don't overlap)"""
    # Check if they share a vertical edge
    vertical_edge = (abs(box1['x2'] - box2['x1']) < tolerance or 
                     abs(box1['x1'] - box2['x2']) < tolerance) and \
                    not (box1['y2'] <= box2['y1'] + tolerance or 
                         box2['y2'] <= box1['y1'] + tolerance)
    
    # Check if they share a horizontal edge
    horizontal_edge = (abs(box1['y2'] - box2['y1']) < tolerance or 
                       abs(box1['y1'] - box2['y2']) < tolerance) and \
                      not (box1['x2'] <= box2['x1'] + tolerance or 
                           box2['x2'] <= box1['x1'] + tolerance)
    
    return vertical_edge or horizontal_edge


def classify_pins_from_nets(routing_data: Dict[str, Dict], fets_data: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Automatically classify pins based on net names in the data"""
    
    all_nets = set()
    
    # Collect all net names from routing data
    for routing_name, routing_info in routing_data.items():
        if 'net_name' in routing_info:
            all_nets.add(routing_info['net_name'])
    
    # Collect all net names from FET terminals
    for fet_name, fet_info in fets_data.items():
        for terminal_name, terminal_info in fet_info['terminals'].items():
            all_nets.add(terminal_info['net'])
    
    # Classify nets
    supply_pins = []
    output_pins = []
    signal_pins = []
    intermediate_nets = []
    
    for net in all_nets:
        if net in ['VDD', 'VSS']:
            supply_pins.append(net)
        elif net == 'Y':  # Assuming Y is the output
            output_pins.append(net)
        elif net.startswith('net'):  # Intermediate nets like net13, net29, net30
            intermediate_nets.append(net)
        else:  # Signal inputs like A1, A2, B1, B2
            signal_pins.append(net)
    
    return {
        'signal_pins': sorted(signal_pins),
        'supply_pins': sorted(supply_pins),
        'output_pins': sorted(output_pins),
        'intermediate_nets': sorted(intermediate_nets),
        'all_pins': sorted(signal_pins + supply_pins + output_pins)
    }


def build_routing_connectivity_map(routing_data: Dict[str, Dict], fets_data: Dict[str, Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a comprehensive connectivity map for all routing boxes.
    For each routing box, find all possible connections:
    - Same-layer edge connections
    - Adjacent layer overlapping connections
    - FET terminal connections (for layer 0 routing)
    """
    
    # Organize routing by layer and net
    routing_by_layer = defaultdict(list)
    routing_by_net = defaultdict(list)
    
    for routing_name, routing_info in routing_data.items():
        layer = routing_info['layer']
        net_name = routing_info.get('net_name', 'unknown')
        
        routing_entry = {
            'name': routing_name,
            'layer': layer,
            'box': routing_info['box'],
            'net_name': net_name
        }
        routing_by_layer[layer].append(routing_entry)
        routing_by_net[net_name].append(routing_entry)
    
    # Build connectivity map
    connectivity_map = {}
    
    print("Building routing connectivity map...")
    
    for routing_name, routing_info in routing_data.items():
        current_layer = routing_info['layer']
        current_box = routing_info['box']
        current_net = routing_info.get('net_name', 'unknown')
        
        connections = {
            'same_layer_edge': [],  # Same layer edge connections
            'lower_layer': [],       # Connections to layer below
            'upper_layer': [],       # Connections to layer above
            'fet_terminals': []      # FET connections (for layer 0)
        }
        
        # Find same-layer edge connections (must be same net)
        for other_routing in routing_by_layer[current_layer]:
            if other_routing['name'] != routing_name and \
               other_routing['net_name'] == current_net and \
               boxes_share_edge(current_box, other_routing['box']):
                connections['same_layer_edge'].append(other_routing['name'])
        
        # Find lower layer connections (overlapping boxes of same net)
        if current_layer > 0:
            lower_layer = current_layer - 1
            for lower_routing in routing_by_layer[lower_layer]:
                if lower_routing['net_name'] == current_net and \
                   boxes_overlap(current_box, lower_routing['box']):
                    connections['lower_layer'].append(lower_routing['name'])
        
        # Find upper layer connections (overlapping boxes of same net)
        upper_layer = current_layer + 1
        for upper_routing in routing_by_layer[upper_layer]:
            if upper_routing['net_name'] == current_net and \
               boxes_overlap(current_box, upper_routing['box']):
                connections['upper_layer'].append(upper_routing['name'])
        
        # For layer 0, find FET terminal connections
        if current_layer == 0:
            for fet_name, fet_info in fets_data.items():
                for terminal_name, terminal_info in fet_info['terminals'].items():
                    if terminal_info['net'] == current_net and \
                       boxes_overlap(current_box, terminal_info['box']):
                        connections['fet_terminals'].append({
                            'fet_name': fet_name,
                            'terminal': terminal_name,
                            'fet_type': 'PMOS' if fet_info.get('np', False) else 'NMOS'
                        })
        
        connectivity_map[routing_name] = connections
    
    return connectivity_map


def trace_routing_paths(start_routing: str, target_type: str, connectivity_map: Dict, 
                       routing_data: Dict, direction: str = 'down') -> List[List[str]]:
    """
    Trace all possible routing paths from a starting routing box.
    
    Args:
        start_routing: Starting routing box name
        target_type: 'fet' for downward to FET, 'top' for upward to top layer
        connectivity_map: Pre-computed connectivity map
        routing_data: Original routing data
        direction: 'down' for signal input paths, 'up' for output paths
    
    Returns:
        List of routing paths (each path is a list of routing names)
    """
    
    all_paths = []
    
    def dfs_trace(current: str, path: List[str], visited: Set[str]):
        # Check for circular path
        if current in visited:
            return
        
        # Check if we reached the target
        if target_type == 'fet' and connectivity_map[current]['fet_terminals']:
            # Reached a FET connection
            all_paths.append(path + [current])
            return
        elif target_type == 'top':
            current_layer = routing_data[current]['layer']
            if current_layer == 5:  # Reached top layer
                all_paths.append(path + [current])
                return
        
        # Continue exploring
        visited.add(current)
        new_path = path + [current]
        
        # Explore same-layer edge connections
        for next_routing in connectivity_map[current]['same_layer_edge']:
            if next_routing not in visited:
                dfs_trace(next_routing, new_path, visited.copy())
        
        # Explore adjacent layer connections based on direction
        if direction == 'down':
            # For downward, explore lower layer connections
            for next_routing in connectivity_map[current]['lower_layer']:
                if next_routing not in visited:
                    dfs_trace(next_routing, new_path, visited.copy())
        else:  # direction == 'up'
            # For upward, explore upper layer connections
            for next_routing in connectivity_map[current]['upper_layer']:
                if next_routing not in visited:
                    dfs_trace(next_routing, new_path, visited.copy())
    
    # Start DFS
    dfs_trace(start_routing, [], set())
    
    return all_paths


def extract_signal_paths(routing_data: Dict[str, Dict], fets_data: Dict[str, Dict], 
                        pin_info: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Extract complete signal paths using flexible routing algorithm"""
    
    print("\n=== SIGNAL PATH EXTRACTION (Enhanced) ===")
    
    # Build connectivity map
    connectivity_map = build_routing_connectivity_map(routing_data, fets_data)
    
    # Organize routing by net and layer
    routing_by_net = defaultdict(list)
    for routing_name, routing_info in routing_data.items():
        net_name = routing_info.get('net_name', 'unknown')
        routing_by_net[net_name].append({
            'name': routing_name,
            'layer': routing_info['layer'],
            'box': routing_info['box']
        })
    
    # Create FET terminal lookup
    fet_terminals_by_net = defaultdict(list)
    for fet_name, fet_info in fets_data.items():
        fet_type = 'PMOS' if fet_info.get('np', False) else 'NMOS'
        for terminal_name, terminal_info in fet_info['terminals'].items():
            net = terminal_info['net']
            fet_terminals_by_net[net].append({
                'fet_name': fet_name,
                'terminal': terminal_name,
                'net': net,
                'box': terminal_info['box'],
                'fet_type': fet_type
            })
    
    complete_signal_paths = []
    
    # For each signal pin, find complete paths to output Y
    for signal_pin in pin_info['signal_pins']:
        print(f"\nTracing paths from signal pin: {signal_pin}")
        
        # Find top layer (layer 5) routing for this signal
        signal_routing = routing_by_net.get(signal_pin, [])
        layer5_signal_boxes = [r for r in signal_routing if r['layer'] == 5]
        
        for start_box in layer5_signal_boxes:
            # Trace downward paths from this starting point
            input_paths = trace_routing_paths(
                start_box['name'], 
                'fet', 
                connectivity_map, 
                routing_data, 
                'down'
            )
            
            for input_path in input_paths:
                # Get the FET connection at the end of this path
                last_routing = input_path[-1]
                fet_connections = connectivity_map[last_routing]['fet_terminals']
                
                for fet_conn in fet_connections:
                    input_fet_name = fet_conn['fet_name']
                    input_fet_terminal = fet_conn['terminal']
                    input_fet_type = fet_conn['fet_type']
                    
                    print(f"  Input path: {signal_pin} → {' → '.join(input_path)} → {input_fet_name}.{input_fet_terminal}")
                    
                    # Now find connections from this FET to output
                    input_fet_info = fets_data[input_fet_name]
                    
                    for terminal_name, terminal_info in input_fet_info['terminals'].items():
                        if terminal_name == 'G':  # Skip gate
                            continue
                        
                        ds_net = terminal_info['net']
                        
                        # Skip supply nets and same signal
                        if ds_net in pin_info['supply_pins'] or ds_net == signal_pin:
                            continue
                        
                        if ds_net in pin_info['intermediate_nets']:
                            # DS-sharing path
                            print(f"    DS-sharing via {ds_net}")
                            
                            # Find other FETs connected to this intermediate net
                            for other_fet_terminal in fet_terminals_by_net.get(ds_net, []):
                                if other_fet_terminal['fet_name'] == input_fet_name:
                                    continue
                                
                                output_fet_name = other_fet_terminal['fet_name']
                                output_fet_info = fets_data[output_fet_name]
                                
                                # Check if this FET connects to Y
                                for y_terminal_name, y_terminal_info in output_fet_info['terminals'].items():
                                    if y_terminal_info['net'] == 'Y':
                                        # Find upward routing paths from this FET to Y
                                        output_paths = find_output_routing_paths_enhanced(
                                            output_fet_name, 
                                            y_terminal_name, 
                                            connectivity_map, 
                                            routing_data
                                        )
                                        
                                        for output_path in output_paths:
                                            complete_path = {
                                                'signal_pin': signal_pin,
                                                'output_pin': 'Y',
                                                'input_routing_path': input_path,
                                                'input_fet': {
                                                    'name': input_fet_name,
                                                    'terminal': input_fet_terminal,
                                                    'type': input_fet_type
                                                },
                                                'ds_sharing': {
                                                    'shared_net': ds_net,
                                                    'input_fet_terminal': terminal_name,
                                                    'output_fet_terminal': other_fet_terminal['terminal']
                                                },
                                                'output_fet': {
                                                    'name': output_fet_name,
                                                    'terminal': y_terminal_name,
                                                    'type': other_fet_terminal['fet_type']
                                                },
                                                'output_routing_path': output_path,
                                                'complete_path': [signal_pin] + input_path + 
                                                    [f"{input_fet_name}.{input_fet_terminal}",
                                                     f"{input_fet_name}.{terminal_name}",
                                                     f"DS-sharing({ds_net})",
                                                     f"{output_fet_name}.{other_fet_terminal['terminal']}",
                                                     f"{output_fet_name}.{y_terminal_name}"] + 
                                                    output_path + ['Y']
                                            }
                                            complete_signal_paths.append(complete_path)
                                            print(f"    COMPLETE PATH: {' → '.join(complete_path['complete_path'])}")
                        
                        elif ds_net == 'Y':
                            # Direct connection to Y
                            print(f"    Direct connection to Y")
                            
                            output_paths = find_output_routing_paths_enhanced(
                                input_fet_name, 
                                terminal_name, 
                                connectivity_map, 
                                routing_data
                            )
                            
                            for output_path in output_paths:
                                complete_path = {
                                    'signal_pin': signal_pin,
                                    'output_pin': 'Y',
                                    'input_routing_path': input_path,
                                    'input_fet': {
                                        'name': input_fet_name,
                                        'terminal': input_fet_terminal,
                                        'type': input_fet_type
                                    },
                                    'direct_output_connection': {
                                        'fet_terminal': terminal_name
                                    },
                                    'output_routing_path': output_path,
                                    'complete_path': [signal_pin] + input_path + 
                                        [f"{input_fet_name}.{input_fet_terminal}",
                                         f"{input_fet_name}.{terminal_name}"] + 
                                        output_path + ['Y']
                                }
                                complete_signal_paths.append(complete_path)
                                print(f"    COMPLETE PATH: {' → '.join(complete_path['complete_path'])}")
    
    return complete_signal_paths


def find_output_routing_paths_enhanced(fet_name: str, terminal_name: str, 
                                      connectivity_map: Dict, routing_data: Dict) -> List[List[str]]:
    """Find upward routing paths from FET to Y using flexible routing"""
    
    # Find layer 0 routing boxes connected to this FET terminal
    starting_routes = []
    
    for routing_name, connections in connectivity_map.items():
        if routing_data[routing_name]['layer'] == 0:
            for fet_conn in connections['fet_terminals']:
                if fet_conn['fet_name'] == fet_name and fet_conn['terminal'] == terminal_name:
                    starting_routes.append(routing_name)
    
    all_output_paths = []
    
    # Trace upward from each starting route
    for start_routing in starting_routes:
        output_paths = trace_routing_paths(
            start_routing, 
            'top', 
            connectivity_map, 
            routing_data, 
            'up'
        )
        all_output_paths.extend(output_paths)
    
    return all_output_paths


def extract_supply_paths(routing_data: Dict[str, Dict], fets_data: Dict[str, Dict], 
                        supplies_data: Dict[str, Any], pin_info: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Extract supply paths using enhanced routing algorithm"""
    
    print("\n=== SUPPLY PATH EXTRACTION (Enhanced) ===")
    
    # Build connectivity map
    connectivity_map = build_routing_connectivity_map(routing_data, fets_data)
    
    # Organize routing by net
    routing_by_net = defaultdict(list)
    for routing_name, routing_info in routing_data.items():
        net_name = routing_info.get('net_name', 'unknown')
        routing_entry = {
            'name': routing_name,
            'layer': routing_info['layer'],
            'box': routing_info['box'],
            'net_name': net_name
        }
        routing_by_net[net_name].append(routing_entry)
    
    supply_paths = []
    
    # Get supply rail information
    supplies = supplies_data.get('supplies', {})
    identified_supplies = supplies_data.get('identified', {})
    
    print(f"Available supplies: {list(supplies.keys())}")
    print(f"Identified supplies: {identified_supplies}")
    
    # Separate layer 16 (power rails) and layer 17 (intermediate supplies)
    layer16_supplies = {name: info for name, info in supplies.items() if info.get('layer') == 16}
    layer17_supplies = {name: info for name, info in supplies.items() if info.get('layer') == 17}
    
    print(f"Layer 16 supplies (power rails): {list(layer16_supplies.keys())}")
    print(f"Layer 17 supplies: {list(layer17_supplies.keys())}")
    
    # Find Layer 16 → Layer 17 connections
    layer16_to_17_connections = []
    for l16_name, l16_info in layer16_supplies.items():
        l16_type = l16_info.get('tag')  # VDD or VSS
        
        for l17_name, l17_info in layer17_supplies.items():
            if boxes_overlap(l16_info['box'], l17_info['box']):
                layer16_to_17_connections.append({
                    'layer16_supply': l16_name,
                    'layer17_supply': l17_name,
                    'supply_type': l16_type
                })
                print(f"  Layer 16→17 connection: {l16_name} → {l17_name} ({l16_type})")
    
    # Process each Layer 16→17 connection
    for connection in layer16_to_17_connections:
        l16_supply = connection['layer16_supply']
        l17_supply = connection['layer17_supply']
        supply_type = connection['supply_type']
        
        l17_info = layer17_supplies[l17_supply]
        
        # Find FET connections
        for fet_name, fet_info in fets_data.items():
            fet_type = 'PMOS' if fet_info.get('np', False) else 'NMOS'
            
            for terminal_name, terminal_info in fet_info['terminals'].items():
                terminal_net = terminal_info['net']
                
                # Check supply connection
                supply_connection = False
                if terminal_net == supply_type:
                    supply_connection = True
                elif supply_type == 'VDD' and fet_type == 'PMOS' and terminal_name == 'S':
                    supply_connection = True
                elif supply_type == 'VSS' and fet_type == 'NMOS' and terminal_name == 'S':
                    supply_connection = True
                
                if supply_connection and boxes_overlap(l17_info['box'], terminal_info['box']):
                    print(f"  Layer 17→FET: {l17_supply} → {fet_name}.{terminal_name}")
                    
                    # Find paths from this FET to output
                    fet_terminals = fet_info['terminals']
                    for y_terminal_name, y_terminal_info in fet_terminals.items():
                        if y_terminal_name == terminal_name:
                            continue
                        
                        y_net = y_terminal_info['net']
                        
                        if y_net == 'Y':
                            # Direct connection to Y
                            output_paths = find_output_routing_paths_enhanced(
                                fet_name, 
                                y_terminal_name, 
                                connectivity_map, 
                                routing_data
                            )
                            
                            for output_path in output_paths:
                                supply_path = {
                                    'start_supply': supply_type,
                                    'supply_chain': [l16_supply, l17_supply],
                                    'fet_connection': {
                                        'fet_name': fet_name,
                                        'supply_terminal': terminal_name,
                                        'output_terminal': y_terminal_name,
                                        'fet_type': fet_type
                                    },
                                    'output_routing_path': output_path,
                                    'complete_path': [l16_supply, l17_supply, 
                                        f"{fet_name}.{terminal_name}",
                                        f"{fet_name}.{y_terminal_name}"] + 
                                        output_path + ['Y'],
                                    'network_type': 'pull_up' if supply_type == 'VDD' else 'pull_down'
                                }
                                supply_paths.append(supply_path)
                                print(f"    COMPLETE SUPPLY PATH: {' → '.join(supply_path['complete_path'])}")
                        
                        elif y_net in pin_info['intermediate_nets']:
                            # DS-sharing connection
                            print(f"    DS-sharing via {y_net}")
                            
                            # Find other FETs sharing this net
                            for other_fet_name, other_fet_info in fets_data.items():
                                if other_fet_name == fet_name:
                                    continue
                                
                                for other_terminal_name, other_terminal_info in other_fet_info['terminals'].items():
                                    if other_terminal_info['net'] == y_net:
                                        # Check if the other FET connects to Y
                                        for final_terminal_name, final_terminal_info in other_fet_info['terminals'].items():
                                            if final_terminal_info['net'] == 'Y':
                                                # Find routing paths to Y
                                                output_paths = find_output_routing_paths_enhanced(
                                                    other_fet_name, 
                                                    final_terminal_name, 
                                                    connectivity_map, 
                                                    routing_data
                                                )
                                                
                                                for output_path in output_paths:
                                                    supply_path = {
                                                        'start_supply': supply_type,
                                                        'supply_chain': [l16_supply, l17_supply],
                                                        'fet_connection': {
                                                            'fet_name': fet_name,
                                                            'supply_terminal': terminal_name,
                                                            'output_terminal': y_terminal_name,
                                                            'fet_type': fet_type
                                                        },
                                                        'ds_sharing': {
                                                            'shared_net': y_net,
                                                            'first_fet_terminal': y_terminal_name,
                                                            'second_fet_terminal': other_terminal_name
                                                        },
                                                        'output_fet': {
                                                            'fet_name': other_fet_name,
                                                            'terminal': final_terminal_name,
                                                            'fet_type': 'PMOS' if other_fet_info.get('np', False) else 'NMOS'
                                                        },
                                                        'output_routing_path': output_path,
                                                        'complete_path': [l16_supply, l17_supply,
                                                            f"{fet_name}.{terminal_name}",
                                                            f"{fet_name}.{y_terminal_name}",
                                                            f"DS-sharing({y_net})",
                                                            f"{other_fet_name}.{other_terminal_name}",
                                                            f"{other_fet_name}.{final_terminal_name}"] +
                                                            output_path + ['Y'],
                                                        'network_type': 'pull_up' if supply_type == 'VDD' else 'pull_down'
                                                    }
                                                    supply_paths.append(supply_path)
                                                    print(f"    COMPLETE SUPPLY PATH: {' → '.join(supply_path['complete_path'])}")
    
    return supply_paths


def create_comprehensive_topology(input_file: str, output_file: str) -> Dict[str, Any]:
    """Create comprehensive topology combining signal and supply path extraction"""
    
    print("Loading processed layout data...")
    with open(input_file, 'r') as f:
        layout_data = json.load(f)
    
    # Extract data sections
    routing_data = layout_data['routing']
    fets_data = layout_data['fets']
    supplies_data = layout_data['layout']
    
    print(f"Data summary:")
    print(f"  Routing boxes: {len(routing_data)}")
    print(f"  FETs: {len(fets_data)}")
    print(f"  Supply rails: {len(supplies_data.get('supplies', {}))}")
    
    # Classify pins automatically
    print("\nClassifying pins from net names...")
    pin_info = classify_pins_from_nets(routing_data, fets_data)
    print(f"Pin classification:")
    for pin_type, pins in pin_info.items():
        print(f"  {pin_type}: {pins}")
    
    # Extract signal paths with enhanced algorithm
    print("\n" + "="*50)
    signal_paths = extract_signal_paths(routing_data, fets_data, pin_info)
    
    # Extract supply paths with enhanced algorithm
    print("\n" + "="*50)
    supply_paths = extract_supply_paths(routing_data, fets_data, supplies_data, pin_info)
    
    # Create comprehensive output
    output_data = {
        "signal_paths": signal_paths,
        "supply_paths": supply_paths
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n" + "="*50)
    print(f"=== EXTRACTION COMPLETE ===")
    print(f"Signal paths found: {len(signal_paths)}")
    print(f"Supply paths found: {len(supply_paths)}")
    print(f"Results saved to: {output_file}")
    
    return output_data


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python path_extractor_enhanced.py <processed_raw.json> [output.json]")
        print("Example: python path_extractor_enhanced.py processed_raw.json topology.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clean_topology.json"
    
    try:
        result = create_comprehensive_topology(input_file, output_file)
        
        # Print summary
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Signal Paths: {len(result['signal_paths'])}")
        print(f"Supply Paths: {len(result['supply_paths'])}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()