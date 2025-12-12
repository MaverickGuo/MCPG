#!/usr/bin/env python3

import json
import numpy as np
from collections import defaultdict


def _make_printer(verbose: bool):
    def _printer(*args, force: bool = False, **kwargs):
        if verbose or force:
            print(*args, **kwargs)

    return _printer

class NodeFeatureGenerator:
    def __init__(self, net_file: str, processed_file: str, topology_file: str):
        self.net_file = net_file
        self.processed_file = processed_file
        self.topology_file = topology_file
        
        self.net_data = self._parse_netlist()
        self.layout_data = self._load_json(processed_file)
        self.topology_data = self._load_json(topology_file)
        
        self.net_to_id = self._build_net_mapping()
        
    def _load_json(self, filepath: str):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _parse_netlist(self):
        net_data = {'pins': [], 'instances': {}}
        
        with open(self.net_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('.SUBCKT'):
                    parts = line.split()
                    net_data['pins'] = parts[2:]  # Skip .SUBCKT and cell_name
                elif line.startswith('MM'):
                    parts = line.split()
                    inst_name = parts[0]
                    drain, gate, source, bulk = parts[1:5]
                    device_type = parts[5]
                    
                    # Parse parameters
                    params = {}
                    for param in parts[6:]:
                        if '=' in param:
                            key, val = param.split('=')
                            if key in ['w', 'l']:
                                params[key] = float(val.replace('n', 'e-9'))
                            elif key == 'nfin':
                                params[key] = int(val)
                    
                    net_data['instances'][inst_name] = {
                        'terminals': {'D': drain, 'G': gate, 'S': source, 'B': bulk},
                        'type': device_type,
                        'params': params
                    }
        
        return net_data
    
    def _build_net_mapping(self):
        net_to_id = {}
        
        # Primary pins first
        for i, pin in enumerate(self.net_data['pins']):
            net_to_id[pin] = i
        
        # Internal nets
        all_nets = set()
        for inst_data in self.net_data['instances'].values():
            for net in inst_data['terminals'].values():
                all_nets.add(net)
        
        next_id = len(self.net_data['pins'])
        for net in sorted(all_nets):
            if net not in net_to_id:
                net_to_id[net] = next_id
                next_id += 1
                
        return net_to_id
    
    def _get_electrical_attribute(self, net_name: str):
        # Check pull-up/pull-down from topology
        for path in self.topology_data.get('supply_paths', []):
            if path.get('network_type') == 'pull_up':
                if net_name in path.get('complete_path', []):
                    return [1, 1, 0]  # pull_up
            elif path.get('network_type') == 'pull_down':
                if net_name in path.get('complete_path', []):
                    return [1, 0, 1]  # pull_down
        return [0, 0, 0]  # neither
    
    def _get_signal_type(self, net_name: str):
        if net_name in ['A1', 'A2', 'B1', 'B2']:
            return [1, 1, 0]  # signal input
        elif net_name == 'Y':
            return [1, 0, 1]  # signal output
        return [0, 0, 0]  # not signal
    
    def generate_routing_features(self):
        routing_features = []
        
        for routing_id, routing_data in self.layout_data.get('routing', {}).items():
            # Type tag: [1,0,0] for routing
            type_tag = [1, 0, 0]
            
            # Electrical features (without hop distances)
            layer = routing_data['layer']
            net_name = routing_data['net_name']
            net_id = self.net_to_id.get(net_name, -1)
            elec_attr = self._get_electrical_attribute(net_name)  # [3] one-hot
            signal_type = self._get_signal_type(net_name)  # [3] one-hot
            
            electrical_features = [
                layer, net_id
            ] + elec_attr + signal_type
            
            # Physical features
            box = routing_data['box']
            bbox_features = [box['x1'], box['x2'], box['y1'], box['y2']]
            lat_cpl = routing_data.get('lat_cpl', 0.0)
            ver_cpl = routing_data.get('ver_cpl', 0.0)
            
            physical_features = bbox_features + [lat_cpl, ver_cpl]
            
            # Combine all features
            feature_vector = np.array(type_tag + electrical_features + physical_features, dtype=np.float32)
            routing_features.append(feature_vector)
        
        return routing_features
    
    def generate_fet_features(self):
        fet_features = []
        
        for fet_name, fet_data in self.layout_data.get('fets', {}).items():
            # Type tag: [0,1,0] for FET
            type_tag = [0, 1, 0]
            
            # Get netlist parameters
            netlist_fet = self.net_data['instances'].get(fet_name, {})
            params = netlist_fet.get('params', {})
            
            # Electrical features (without hop distances)
            w = params.get('w', 0.0)
            l = params.get('l', 0.0)
            nfin = params.get('nfin', 0)
            
            # Device type: PMOS=1, NMOS=-1
            np_val = 1 if fet_data.get('np', True) else -1
            
            # Terminal connections
            terminals = fet_data.get('terminals', {})
            term_conn_g = self.net_to_id.get(terminals.get('G', {}).get('net', ''), -1)
            term_conn_d = self.net_to_id.get(terminals.get('D', {}).get('net', ''), -1)
            term_conn_s = self.net_to_id.get(terminals.get('S', {}).get('net', ''), -1)
            
            # Get gate net for other features
            gate_net = terminals.get('G', {}).get('net', '')
            elec_attr = self._get_electrical_attribute(gate_net)  # [3] one-hot
            signal_type = self._get_signal_type(gate_net)  # [3] one-hot
            
            electrical_features = [
                w, l, nfin, np_val
            ] + elec_attr + signal_type + [
                term_conn_g, term_conn_d, term_conn_s
            ]
            
            # Physical features
            xs = []
            ys = []
            for terminal_data in terminals.values():
                bbox = terminal_data.get('box', {})
                if bbox:
                    xs.extend([bbox.get('x1', 0), bbox.get('x2', 0)])
                    ys.extend([bbox.get('y1', 0), bbox.get('y2', 0)])

            if xs and ys:
                whole_bbox = [min(xs), max(xs), min(ys), max(ys)]
            else:
                whole_bbox = [0, 0, 0, 0]

            physical_features = whole_bbox

            # Flip
            flip = float(fet_data.get('location', {}).get('flip', False))
            physical_features.append(flip)
            
            # Combine all features
            feature_vector = np.array(type_tag + electrical_features + physical_features, dtype=np.float32)
            fet_features.append(feature_vector)
        
        return fet_features
    
    def generate_supply_features(self):
        supply_features = []
        
        supplies = self.layout_data.get('layout', {}).get('supplies', {})
        identified = self.layout_data.get('layout', {}).get('identified', {})

        for supply_id, supply_data in supplies.items():
            # Type tag: [0,0,1] for supply
            type_tag = [0, 0, 1]
            
            # Electrical features (without hop distances)
            layer = supply_data.get('layer', 0)
            
            # Determine supply role for electrical attribute encoding
            elec_attr = [0, 0, 0]
            supply_tag = supply_data.get('tag', '')
            if supply_tag in ['VDD', 'VSS']:
                elec_attr = [1, 1, 0] if supply_tag == 'VDD' else [1, 0, 1]
            else:
                # Check if this supply is identified as VDD/VSS
                for net_name, supply_ref in identified.items():
                    if supply_ref == supply_id and net_name in ['VDD', 'VSS']:
                        elec_attr = [1, 1, 0] if net_name == 'VDD' else [1, 0, 1]
                        break
            
            electrical_features = [layer] + elec_attr
            
            # Physical features
            box = supply_data.get('box', {})
            bbox_features = [box.get('x1', 0), box.get('x2', 0), 
                           box.get('y1', 0), box.get('y2', 0)]
            width = supply_data.get('width', 0.0)
            height = supply_data.get('height', 0.0)
            
            physical_features = bbox_features + [width, height]
            
            # Combine all features
            feature_vector = np.array(type_tag + electrical_features + physical_features, dtype=np.float32)
            supply_features.append(feature_vector)

        return supply_features
    
    def generate_all_features(self):
        return {
            'routing': self.generate_routing_features(),
            'fet': self.generate_fet_features(),
            'supply': self.generate_supply_features()
        }


class HeteroGraphBuilder:
    def __init__(self, net_file: str, processed_file: str, topology_file: str, verbose: bool = False):
        self.net_file = net_file
        self.processed_file = processed_file
        self.topology_file = topology_file

        self.layout_data = self._load_json(processed_file)
        self.topology_data = self._load_json(topology_file)

        self._log = _make_printer(verbose)

        # 节点映射
        self.node_to_id = {}
        self.id_to_node = {}
        self._build_node_mapping()
        
        # Edge类型定义
        self.edge_types = {
            'routing-routing': 0,
            'input-routing': 1, 
            'routing-output': 2,
            'routing-g': 3,
            'routing-d': 4, 
            'routing-s': 5,
            'ds-sharing': 6,
            'vdd-routing': 7,
            'vss-routing': 8
        }
        
    def _load_json(self, filepath: str):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _build_node_mapping(self):
        """构建统一的节点映射"""
        node_id = 0
        
        # 1. Routing nodes
        for routing_id in self.layout_data.get('routing', {}):
            self.node_to_id[routing_id] = node_id
            self.id_to_node[node_id] = routing_id
            node_id += 1
        
        # 2. FET nodes (以FET名称为节点，不区分terminal)
        for fet_name in self.layout_data.get('fets', {}):
            self.node_to_id[fet_name] = node_id
            self.id_to_node[node_id] = fet_name
            node_id += 1
        
        # 3. Supply nodes
        supplies = self.layout_data.get('layout', {}).get('supplies', {})
        for supply_id in supplies:
            self.node_to_id[supply_id] = node_id
            self.id_to_node[node_id] = supply_id
            node_id += 1

        self._log(f"总共 {node_id} 个节点")
        self._log(f"Routing: {len(self.layout_data.get('routing', {}))}")
        self._log(f"FET: {len(self.layout_data.get('fets', {}))}")
        self._log(f"Supply: {len(supplies)}")
    
    def _get_physical_distance(self, node1: str, node2: str) -> float:
        """计算两个节点之间的物理距离"""
        def get_node_center(node):
            # Routing节点
            if node.startswith('routing_') and node in self.layout_data.get('routing', {}):
                box = self.layout_data['routing'][node]['box']
                return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)
            
            # FET节点 - 使用整体bounding box中心
            if node in self.layout_data.get('fets', {}):
                terminals = self.layout_data['fets'][node]['terminals']
                all_coords = []
                for terminal_data in terminals.values():
                    bbox = terminal_data.get('box', {})
                    if bbox:
                        all_coords.extend([bbox.get('x1', 0), bbox.get('x2', 0)])
                        all_coords.extend([bbox.get('y1', 0), bbox.get('y2', 0)])
                if all_coords:
                    center_x = (min(all_coords[::2]) + max(all_coords[::2])) / 2
                    center_y = (min(all_coords[1::2]) + max(all_coords[1::2])) / 2
                    return (center_x, center_y)
            
            # Supply节点
            supplies = self.layout_data.get('layout', {}).get('supplies', {})
            if node in supplies:
                box = supplies[node]['box']
                return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)
            
            return (0, 0)  # 简化处理
        
        try:
            center1 = get_node_center(node1)
            center2 = get_node_center(node2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            return float(distance)
        except:
            return 0.0
    
    def _get_electrical_strength(self, edge_type: str) -> float:
        """根据edge类型返回电气连接强度"""
        strength_map = {
            'routing-routing': 1.0,
            'input-routing': 1.0,
            'routing-output': 1.0,
            'routing-g': 0.8,    # Gate连接阻抗较高
            'routing-d': 1.0,    # Drain连接
            'routing-s': 1.0,    # Source连接
            'ds-sharing': 0.9,   # DS sharing连接
            'vdd-routing': 1.0,  # Supply连接
            'vss-routing': 1.0
        }
        return strength_map.get(edge_type, 0.5)
    
    def build_graph_from_topology(self):
        """从topology构建异构图"""
        edges = []
        edge_types = []
        edge_features = []
        
        # 1. 处理Signal Paths
        for signal_path in self.topology_data.get('signal_paths', []):
            path = signal_path.get('complete_path', [])
            path_context = {
                'type': 'signal',
                'signal_pin': signal_path.get('signal_pin'),
                'output_pin': signal_path.get('output_pin'),
                'full_path_data': signal_path
            }
            self._extract_edges_from_path(path, edges, edge_types, edge_features, path_context=path_context)
        
        # 2. 处理Supply Paths
        for supply_path in self.topology_data.get('supply_paths', []):
            path = supply_path.get('complete_path', [])
            start_supply = supply_path.get('start_supply')
            if start_supply and path:
                full_path = [start_supply] + path
                path_context = {
                    'type': 'supply',
                    'network_type': supply_path.get('network_type'),
                    'start_supply': start_supply,
                    'full_path_data': supply_path
                }
                self._extract_edges_from_path(full_path, edges, edge_types, edge_features, path_context=path_context)
        
        # 转换为numpy数组，避免对torch的依赖
        feature_dim = len(edge_features[0]) if edge_features else 17
        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
            edge_type_array = np.array(edge_types, dtype=np.int64)
            edge_feature_array = np.array(edge_features, dtype=np.float32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_type_array = np.empty(0, dtype=np.int64)
            edge_feature_array = np.empty((0, feature_dim), dtype=np.float32)

        edge_count = edge_index.shape[1] if edge_index.size else 0
        type_distribution = (
            np.bincount(edge_type_array).tolist() if edge_type_array.size else []
        )

        self._log(f"构建了 {edge_count} 条边")
        self._log(f"Edge types 分布: {type_distribution}")

        return edge_index, edge_type_array, edge_feature_array
    
    def _extract_edges_from_path(self, path, edges, edge_types, edge_features, path_context=None):
        """从单个path中提取edges"""
        for i in range(len(path) - 1):
            current = path[i]
            next_elem = path[i + 1]
            
            # 跳过DS-sharing，但记录特殊连接
            if 'DS-sharing' in current:
                continue
            if 'DS-sharing' in next_elem:
                # 寻找DS-sharing后的元素
                if i + 2 < len(path):
                    after_ds = path[i + 2]
                    is_supply = path_context and path_context.get('type') == 'supply'
                    edge_type, src_id, dst_id = self._classify_edge(current, after_ds, path, is_supply)
                    if edge_type is not None:
                        edges.append([src_id, dst_id])
                        edge_types.append(self.edge_types['ds-sharing'])
                        features = self._create_edge_features('ds-sharing', current, after_ds, path_context)
                        edge_features.append(features)
                continue
            
            # 正常的相邻连接
            is_supply = path_context and path_context.get('type') == 'supply'
            edge_type, src_id, dst_id = self._classify_edge(current, next_elem, path, is_supply)
            if edge_type is not None:
                edges.append([src_id, dst_id])
                edge_types.append(self.edge_types[edge_type])
                features = self._create_edge_features(edge_type, current, next_elem, path_context)
                edge_features.append(features)
    
    def _classify_edge(self, node1: str, node2: str, path: list, is_supply: bool):
        """分类edge类型并返回节点ID"""
        # 获取实际的节点名称（去除terminal后缀）
        def get_base_node(node):
            if '.' in node and not node.startswith('DS-sharing'):
                return node.split('.')[0]  # MM2.G -> MM2
            return node
        
        def is_top_routing_with_input_net(node):
            """检查是否是带有input net的top-routing"""
            if node.startswith('routing_') and node in self.layout_data.get('routing', {}):
                net_name = self.layout_data['routing'][node]['net_name']
                return net_name in ['A1', 'A2', 'B1', 'B2']
            return False
        
        def is_top_routing_with_output_net(node):
            """检查是否是带有output net的top-routing"""
            if node.startswith('routing_') and node in self.layout_data.get('routing', {}):
                net_name = self.layout_data['routing'][node]['net_name']
                return net_name == 'Y'
            return False
        
        def is_supply_with_tag(node):
            """检查是否是有tag的supply"""
            supplies = self.layout_data.get('layout', {}).get('supplies', {})
            identified = self.layout_data.get('layout', {}).get('identified', {})
            
            if node in supplies:
                # 直接有tag
                supply_tag = supplies[node].get('tag', '')
                if supply_tag in ['VDD', 'VSS']:
                    return supply_tag
                
                # 通过identified映射查找
                for net_name, supply_ref in identified.items():
                    if supply_ref == node and net_name in ['VDD', 'VSS']:
                        return net_name
            
            return None
        
        base_node1 = get_base_node(node1)
        base_node2 = get_base_node(node2)
        
        # 检查节点是否在映射中
        if base_node1 not in self.node_to_id or base_node2 not in self.node_to_id:
            return None, None, None
        
        src_id = self.node_to_id[base_node1]
        dst_id = self.node_to_id[base_node2]
        
        # 分类edge类型
        
        # 1. VDD/VSS-routing: 有tag的supply + 连接的routing
        supply_tag1 = is_supply_with_tag(base_node1)
        supply_tag2 = is_supply_with_tag(base_node2)
        
        if supply_tag1 and base_node2.startswith('routing_'):
            return 'vdd-routing' if supply_tag1 == 'VDD' else 'vss-routing', src_id, dst_id
        if supply_tag2 and base_node1.startswith('routing_'):
            return 'vdd-routing' if supply_tag2 == 'VDD' else 'vss-routing', dst_id, src_id
        
        # 2. Input-routing: top-routing with input net + 下一层routing
        if is_top_routing_with_input_net(base_node1) and base_node2.startswith('routing_'):
            return 'input-routing', src_id, dst_id
        if is_top_routing_with_input_net(base_node2) and base_node1.startswith('routing_'):
            return 'input-routing', dst_id, src_id
        
        # 3. Routing-output: routing + top-routing with output net
        if base_node1.startswith('routing_') and is_top_routing_with_output_net(base_node2):
            return 'routing-output', src_id, dst_id
        if base_node2.startswith('routing_') and is_top_routing_with_output_net(base_node1):
            return 'routing-output', dst_id, src_id
        
        # 4. FET terminal connections
        if '.' in node1:
            terminal = node1.split('.')[1]
            if terminal == 'G':
                return 'routing-g', dst_id, src_id  # routing -> FET
            elif terminal == 'D':
                return 'routing-d', dst_id, src_id
            elif terminal == 'S':
                return 'routing-s', dst_id, src_id
        
        if '.' in node2:
            terminal = node2.split('.')[1]
            if terminal == 'G':
                return 'routing-g', src_id, dst_id  # routing -> FET
            elif terminal == 'D':
                return 'routing-d', src_id, dst_id
            elif terminal == 'S':
                return 'routing-s', src_id, dst_id
        
        # 5. Routing-routing connections
        if base_node1.startswith('routing_') and base_node2.startswith('routing_'):
            return 'routing-routing', src_id, dst_id
        
        return None, None, None
    
    def _get_path_signal_type(self, path_context):
        """基于path上下文获取signal类型编码"""
        if not path_context:
            return [0, 0, 0]  # no path context
        
        if path_context.get('type') == 'signal':
            signal_pin = path_context.get('signal_pin')
            output_pin = path_context.get('output_pin')
            
            if signal_pin in ['A1', 'A2', 'B1', 'B2']:
                return [1, 1, 0]  # signal input path
            elif output_pin == 'Y':
                return [1, 0, 1]  # signal output path
        
        return [0, 0, 0]  # not signal path
    
    def _get_path_supply_type(self, path_context):
        """基于path上下文获取supply类型编码"""
        if not path_context:
            return [0, 0, 0]  # no path context
        
        if path_context.get('type') == 'supply':
            network_type = path_context.get('network_type')
            
            if network_type == 'pull_up':
                return [1, 1, 0]  # pull_up path
            elif network_type == 'pull_down':
                return [1, 0, 1]  # pull_down path
        
        return [0, 0, 0]  # not supply path
    
    def _create_edge_features(self, edge_type: str, node1: str, node2: str, path_context):
        """创建edge特征向量"""
        # One-hot edge type encoding (9维)
        type_onehot = [0] * 9
        if edge_type in self.edge_types:
            type_onehot[self.edge_types[edge_type]] = 1
        
        # Path-based signal/supply encoding (6维: 3+3)
        path_signal_type = self._get_path_signal_type(path_context)  # [3] one-hot
        path_supply_type = self._get_path_supply_type(path_context)  # [3] one-hot
        
        # Additional features (2维)
        physical_distance = self._get_physical_distance(node1, node2)
        electrical_strength = self._get_electrical_strength(edge_type)

        # 总计: 9 + 3 + 3 + 2 = 17维
        return (type_onehot + 
                path_signal_type + path_supply_type + 
                [physical_distance, electrical_strength])


def compute_laplacian_pe(edge_index, num_nodes, pe_dim=8):
    """计算对称归一化拉普拉斯位置编码 (纯numpy实现)"""
    if num_nodes == 0:
        return np.zeros((0, pe_dim))

    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return np.zeros((num_nodes, pe_dim))

    row = edge_index[0]
    col = edge_index[1]

    # 构建无向邻接矩阵（去重）
    pairs = np.concatenate([np.vstack([row, col]), np.vstack([col, row])], axis=1)
    pairs = np.unique(pairs, axis=1)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    adj[pairs[0], pairs[1]] = 1.0

    # 对称归一化拉普拉斯
    degree = adj.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(degree + 1e-8)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt)
    L = np.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    # 特征分解（L为对称正定矩阵，使用numpy.linalg.eigh）
    try:
        eigenvals, eigenvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros((num_nodes, pe_dim))

    # 丢弃第一个特征向量（对应特征值0），并选取所需数量
    vectors = eigenvecs[:, 1 : pe_dim + 1]
    if vectors.shape[1] < pe_dim:
        pad = np.zeros((num_nodes, pe_dim - vectors.shape[1]))
        vectors = np.hstack([vectors, pad])

    # 统一特征向量符号
    for i in range(vectors.shape[1]):
        idx = np.argmax(np.abs(vectors[:, i]))
        if vectors[idx, i] < 0:
            vectors[:, i] *= -1

    # 标准化
    vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-6)
    return vectors


def generate_graph_embeddings(
    net_file: str,
    processed_file: str,
    topology_file: str,
    output_file: str,
    normalize: bool = False,
    verbose: bool = False,
    print_summary: bool = True,
):
    """生成图的所有embedding：node embeddings, edge embeddings, COO格式"""

    log = _make_printer(verbose)

    # 1. 生成Node Features
    log("=== 生成Node Features ===")
    node_generator = NodeFeatureGenerator(net_file, processed_file, topology_file)
    node_features = node_generator.generate_all_features()

    # 2. 生成Edge Features和Graph Structure  
    log("\n=== 生成Edge Features和Graph Structure ===")
    graph_builder = HeteroGraphBuilder(net_file, processed_file, topology_file, verbose=verbose)
    edge_index, edge_types, edge_features = graph_builder.build_graph_from_topology()

    # 3. 计算拉普拉斯位置编码
    log("\n=== 计算拉普拉斯位置编码 ===")
    num_nodes = len(graph_builder.node_to_id)
    laplacian_pe = compute_laplacian_pe(edge_index, num_nodes, pe_dim=8)
    log(f"计算了 {laplacian_pe.shape[0]} 个节点的 {laplacian_pe.shape[1]} 维拉普拉斯PE")

    # 4. 为节点特征添加拉普拉斯PE
    log("\n=== 添加拉普拉斯PE到节点特征 ===")
    node_data = {
        'node_types': {},
        'feature_dims': {},
        'normalization': {'applied': normalize}
    }
    
    # 构建节点类型到全局ID的映射
    node_type_indices = {
        'routing': [],
        'fet': [],
        'supply': []
    }
    
    # 收集每种节点类型对应的全局节点ID
    for node_name, global_id in graph_builder.node_to_id.items():
        if node_name.startswith('routing_'):
            node_type_indices['routing'].append(global_id)
        elif node_name.startswith('MM'):
            node_type_indices['fet'].append(global_id)
        elif node_name.startswith('supply_'):
            node_type_indices['supply'].append(global_id)
    
    total_nodes = 0
    for node_type, feature_list in node_features.items():
        if not feature_list:
            node_data['node_types'][node_type] = []
            node_data['feature_dims'][node_type] = 8  # 只有PE维度
            continue
        
        # Convert to numpy array
        feature_matrix = np.vstack(feature_list)
        total_nodes += len(feature_list)
        
        # 获取对应的拉普拉斯PE
        node_indices = node_type_indices[node_type]
        if len(node_indices) == len(feature_list):
            pe_for_type = laplacian_pe[node_indices]  # [n_nodes, 8]

            # 在特征前面添加拉普拉斯PE
            feature_matrix_with_pe = np.concatenate([pe_for_type, feature_matrix], axis=1)
        else:
            log(f"Warning: 节点数量不匹配 for {node_type}: {len(node_indices)} vs {len(feature_list)}")
            # 如果数量不匹配，添加零PE
            zero_pe = np.zeros((len(feature_list), 8))
            feature_matrix_with_pe = np.concatenate([zero_pe, feature_matrix], axis=1)
        
        # Normalize if requested
        if normalize:
            # 更新skip_cols以考虑前面的8维PE
            pe_cols = list(range(8))  # 前8维是PE，不需要标准化
            
            if node_type == 'routing':
                # PE(8) + routing: type(3) + layer(1) + net_id(1) + elec_attr(3) + signal_type(3) + physical(6)
                skip_cols = pe_cols + [8+i for i in range(3)] + [8+i for i in range(5, 8)] + [8+i for i in range(8, 11)]
            elif node_type == 'fet':
                # PE(8) + fet: type(3) + w,l,nfin,np(4) + elec_attr(3) + signal_type(3) + term_conn(3) + physical(5)
                skip_cols = pe_cols + [8+i for i in range(3)] + [8+i for i in range(7, 10)] + [8+i for i in range(10, 13)]
            elif node_type == 'supply':
                # PE(8) + supply: type(3) + layer(1) + elec_attr(3) + physical(6)
                skip_cols = pe_cols + [8, 9, 10, 12, 13, 14]
            
            # Separate features to normalize and skip
            skip_features = feature_matrix_with_pe[:, skip_cols]
            norm_cols = [i for i in range(feature_matrix_with_pe.shape[1]) if i not in skip_cols]
            features_to_norm = feature_matrix_with_pe[:, norm_cols]
            
            # Normalize features (avoiding division by zero)
            mean_vals = np.mean(features_to_norm, axis=0)
            std_vals = np.std(features_to_norm, axis=0)
            std_vals[std_vals == 0] = 1.0  # Avoid division by zero
            
            normalized_features = (features_to_norm - mean_vals) / std_vals
            
            # Reconstruct feature matrix with proper order
            final_matrix = np.zeros_like(feature_matrix_with_pe)
            final_matrix[:, skip_cols] = skip_features
            final_matrix[:, norm_cols] = normalized_features
            feature_matrix_with_pe = final_matrix
            
            # Store normalization parameters
            node_data[f'{node_type}_normalization'] = {
                'mean': mean_vals.tolist(),
                'std': std_vals.tolist()
            }
        
        # Store features and metadata
        node_data['node_types'][node_type] = feature_matrix_with_pe.tolist()
        node_data['feature_dims'][node_type] = feature_matrix_with_pe.shape[1]

        log(f"{node_type}: {len(feature_list)} nodes, {feature_matrix.shape[1]} -> {feature_matrix_with_pe.shape[1]} features (added 8 PE)")

    log(f"Generated {total_nodes} node features with Laplacian PE")
    
    # 3. 组合所有数据
    output_data = {
        'node_embeddings': node_data,
        'edge_embeddings': {
            'edge_features': edge_features.tolist(),
            'edge_types': edge_types.tolist(),
            'feature_dims': edge_features.shape[1] if edge_features.size > 0 else 17,
            'edge_type_names': {v: k for k, v in graph_builder.edge_types.items()},
            'feature_description': [
                # Edge type one-hot (9维)
                'routing-routing', 'input-routing', 'routing-output',
                'routing-g', 'routing-d', 'routing-s', 'ds-sharing',
                'vdd-routing', 'vss-routing',
                # Path signal type (3维)
                'path_signal_input', 'path_signal_output', 'path_signal_not',
                # Path supply type (3维)
                'path_supply_pullup', 'path_supply_pulldown', 'path_supply_not',
                # Additional features (2维)
                'physical_distance', 'electrical_strength'
            ]
        },
        'coo_format': {
            'edge_index': edge_index.tolist(),
            'num_nodes': len(graph_builder.node_to_id),
            'num_edges': edge_index.shape[1] if edge_index.size > 0 else 0,
            'node_mapping': {
                'node_to_id': graph_builder.node_to_id,
                'id_to_node': {int(k): v for k, v in graph_builder.id_to_node.items()}
            }
        }
    }
    
    # 4. 保存到文件
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    if print_summary:
        _print_summary(output_data, output_file, verbose)

    return output_data


def _print_summary(output_data, output_file, verbose):
    node_types = output_data.get('node_embeddings', {}).get('node_types', {})
    edge_info = output_data.get('edge_embeddings', {})
    coo_info = output_data.get('coo_format', {})

    node_type_count = len(node_types)
    edge_dim = edge_info.get('feature_dims', 0)
    num_edges = coo_info.get('num_edges', 0)
    num_nodes = coo_info.get('num_nodes', 0)

    if verbose:
        printer = _make_printer(True)
        printer(f"\n=== 所有embedding保存到 {output_file} ===")
        printer(f"Node embeddings: {node_type_count} types")
        printer(f"Edge embeddings: {edge_dim} dimensions, {num_edges} edges")
        printer(f"COO format: {num_nodes} nodes, {num_edges} edges")
    else:
        print(f"Node embeddings: {node_type_count} types")
        print(f"Edge embeddings: {edge_dim} dimensions, {num_edges} edges")
        print(f"COO format: {num_nodes} nodes, {num_edges} edges")
        print(f"Output path: {output_file}")


def _find_missing_components(output_data):
    missing = []

    node_types = output_data.get('node_embeddings', {}).get('node_types', {})
    for key in ('routing', 'fet', 'supply'):
        if not node_types.get(key):
            missing.append(key)

    edge_features = output_data.get('edge_embeddings', {}).get('edge_features', [])
    if not edge_features:
        missing.append('edge_features')

    edge_index = output_data.get('coo_format', {}).get('edge_index', [])
    if not edge_index or len(edge_index) < 2 or len(edge_index[0]) == 0:
        missing.append('edge_index')

    return missing


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive graph embeddings (nodes, edges, COO)')
    parser.add_argument('--net', required=True, help='Path to netlist file')
    parser.add_argument('--layout', required=True, help='Path to processed layout JSON')
    parser.add_argument('--topology', required=True, help='Path to topology JSON')
    parser.add_argument('--output', required=True, help='Output file for all embeddings')
    parser.add_argument('--normalize', action='store_true', help='Apply feature normalization')
    parser.add_argument('--all', action='store_true', help='Print full verbose logs')

    args = parser.parse_args()

    verbose = args.all

    output_data = generate_graph_embeddings(
        args.net,
        args.layout,
        args.topology,
        args.output,
        normalize=args.normalize,
        verbose=verbose,
        print_summary=False,
    )

    missing = _find_missing_components(output_data)
    if missing:
        if verbose:
            print(f"Detected empty outputs ({', '.join(missing)}), retrying once.")

        output_data = generate_graph_embeddings(
            args.net,
            args.layout,
            args.topology,
            args.output,
            normalize=args.normalize,
            verbose=verbose,
            print_summary=False,
        )

        missing = _find_missing_components(output_data)
        if missing:
            print(f"Missing outputs after retry: {', '.join(missing)}")
            raise SystemExit(1)

    _print_summary(output_data, args.output, verbose)


if __name__ == '__main__':
    main()
