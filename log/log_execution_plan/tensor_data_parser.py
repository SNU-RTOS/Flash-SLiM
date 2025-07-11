# tensor_data_parser.py

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

class TensorDataParser:
    def __init__(self, json_file_path: str):
        """Initialize parser with JSON file path"""
        self.json_file_path = json_file_path
        self.raw_data = None
        self.tensors = {}  # Organized tensor data
        self.execution_plan = []
        self.tensor_usage = {}
        self.allocation_summary = {}
        self.shared_memory_groups = []  # Groups of tensors sharing memory
        self.tensor_sharing_map = {}  # Maps tensor_id to its shared group

    def format_bytes(self, size: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:3.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def load_data(self) -> None:
        """Load and parse JSON data"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.raw_data = json.load(f)
            
            self._parse_shared_memory_groups()
            self._parse_tensor_data()
            self._parse_execution_plan()
            self._parse_tensor_usage()
            
        except Exception as e:
            raise Exception(f"Error loading JSON data: {str(e)}")

    def _parse_shared_memory_groups(self) -> None:
        """Parse shared memory group information"""
        self.shared_memory_groups = self.raw_data['report_data']['shared_memory_groups']
        
        # Build tensor sharing map for quick lookup
        self.tensor_sharing_map = {}
        for group in self.shared_memory_groups:
            for tensor in group['tensors']:
                self.tensor_sharing_map[tensor['tensor_id']] = {
                    'group_address': group['address'],
                    'group_address_hex': group['address_hex'],
                    'shared_with': [t['tensor_id'] for t in group['tensors'] 
                                  if t['tensor_id'] != tensor['tensor_id']]
                }

    def _parse_tensor_data(self) -> None:
        """Parse tensor information from JSON data"""
        self.allocation_summary = self.raw_data['report_data']['summary']
        
        # Process tensors by type
        for alloc_type, tensors in self.raw_data['report_data']['tensors_by_type'].items():
            for tensor in tensors:
                sharing_info = self.tensor_sharing_map.get(tensor['tensor_id'], {})
                self.tensors[tensor['tensor_id']] = {
                    'address': tensor['address'],
                    'address_hex': tensor['address_hex'],
                    'size': tensor['size'],
                    'data_type': tensor['data_type'],
                    'allocation_type': alloc_type,
                    'usage_count': tensor['usage_count'],
                    'used_by_nodes': tensor['used_by_nodes'],
                    'shape': tensor.get('shape', ''),
                    'shared_with': sharing_info.get('shared_with', [])
                }

    def _parse_execution_plan(self) -> None:
        """Parse execution plan from JSON data"""
        self.execution_plan = self.raw_data['execution_plan']

    def _parse_tensor_usage(self) -> None:
        """Parse tensor usage information"""
        self.tensor_usage = self.raw_data['tensor_usage']

    def get_tensor_info(self, tensor_id: int) -> Optional[Dict]:
        """Get information for a specific tensor"""
        tensor_info = self.tensors.get(tensor_id)
        if not tensor_info:
            return None
            
        # Add shared memory information
        sharing_info = self.tensor_sharing_map.get(tensor_id, {})
        if sharing_info:
            tensor_info['shared_memory_address'] = sharing_info['group_address_hex']
            tensor_info['shared_with'] = sharing_info['shared_with']
            
        return tensor_info

    def get_shared_memory_group(self, tensor_id: int) -> Optional[Dict]:
        """Get shared memory group information for a tensor"""
        for group in self.shared_memory_groups:
            if any(t['tensor_id'] == tensor_id for t in group['tensors']):
                return group
        return None

    def get_node_memory_requirements(self, node_idx: int) -> Optional[Dict]:
        """Calculate memory requirements for a specific node, accounting for shared memory"""
        node = next((n for n in self.execution_plan if n['node_idx'] == node_idx), None)
        if not node:
            return None

        # Track unique memory addresses to avoid counting shared memory multiple times
        unique_addresses = set()
        input_size = 0
        output_size = 0
        
        # Process input tensors
        input_tensors = []
        for tensor_id in node['inputs']:
            tensor = self.get_tensor_info(tensor_id)
            if tensor:
                input_tensors.append(tensor)
                if tensor['address'] not in unique_addresses:
                    input_size += tensor['size']
                    unique_addresses.add(tensor['address'])

        # Process output tensors
        output_tensors = []
        for tensor_id in node['outputs']:
            tensor = self.get_tensor_info(tensor_id)
            if tensor:
                output_tensors.append(tensor)
                if tensor['address'] not in unique_addresses:
                    output_size += tensor['size']
                    unique_addresses.add(tensor['address'])

        return {
            'node_idx': node_idx,
            'operator': node['operator'],
            'input_tensors': node['inputs'],
            'output_tensors': node['outputs'],
            'input_size': input_size,
            'output_size': output_size,
            'total_size': input_size + output_size,
            'unique_memory_addresses': len(unique_addresses)
        }

    def get_memory_sharing_stats(self) -> Dict:
        """Get statistics about memory sharing"""
        total_tensors = len(self.tensors)
        shared_tensors = sum(len(group['tensors']) for group in self.shared_memory_groups)
        unique_shared_addresses = len(self.shared_memory_groups)
        
        # Calculate memory savings
        total_tensor_size = sum(t['size'] for t in self.tensors.values())
        unique_memory_size = total_tensor_size
        for group in self.shared_memory_groups:
            # Subtract sizes of all but one tensor in each group
            unique_memory_size -= sum(t['size'] for t in group['tensors'][1:])
        
        return {
            'total_tensors': total_tensors,
            'shared_tensors': shared_tensors,
            'unique_shared_addresses': unique_shared_addresses,
            'total_tensor_size': total_tensor_size,
            'unique_memory_size': unique_memory_size,
            'memory_saved': total_tensor_size - unique_memory_size
        }

    def print_summary(self) -> None:
        """Print summary of parsed data, including shared memory information"""
        print("\n=== Tensor Data Summary ===")
        print(f"Total tensors: {self.allocation_summary['total_tensors']}")
        print(f"Total size: {self.format_bytes(self.allocation_summary['total_size'])}")
        
        # Print shared memory statistics
        sharing_stats = self.get_memory_sharing_stats()
        print(f"\nShared Memory Statistics:")
        print(f"Tensors sharing memory: {sharing_stats['shared_tensors']}")
        print(f"Unique shared addresses: {sharing_stats['unique_shared_addresses']}")
        print(f"Memory saved through sharing: {self.format_bytes(sharing_stats['memory_saved'])}")
        
        print("\nAllocation Types:")
        for alloc_type, info in self.allocation_summary['allocation_types'].items():
            print(f"{alloc_type}:")
            print(f"  Count: {info['count']}")
            print(f"  Size: {self.format_bytes(info['total_size'])} ({info['percentage']:.1f}%)")
        
        print("\nShared Memory Groups:")
        for group in self.shared_memory_groups:
            print(f"\nAddress {group['address_hex']}:")
            for tensor in group['tensors']:
                tensor_info = self.get_tensor_info(tensor['tensor_id'])
                print(f"  Tensor {tensor['tensor_id']}: "
                      f"{self.format_bytes(tensor['size'])} "
                      f"({tensor_info['data_type']}, {tensor_info['allocation_type']})")
        
        print("\nExecution Plan:")
        print(f"Total nodes: {len(self.execution_plan)}")
        
        # Print first few nodes as example
        print("\nFirst 3 nodes memory requirements:")
        for node in self.execution_plan[:3]:
            req = self.get_node_memory_requirements(node['node_idx'])
            if req:
                print(f"\nNode {req['node_idx']} ({req['operator']}):")
                print(f"  Input tensors: {req['input_tensors']}")
                print(f"  Output tensors: {req['output_tensors']}")
                print(f"  Input size: {self.format_bytes(req['input_size'])}")
                print(f"  Output size: {self.format_bytes(req['output_size'])}")
                print(f"  Total size: {self.format_bytes(req['total_size'])}")
                print(f"  Unique memory addresses: {req['unique_memory_addresses']}")

def main():
    """Example usage of TensorDataParser"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tensor_data_parser.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Parse tensor data
    parser = TensorDataParser(json_file)
    parser.load_data()
    
    # Print summary
    parser.print_summary()
    
    # Example of getting specific tensor info
    tensor_id = list(parser.tensors.keys())[0]  # Get first tensor
    tensor_info = parser.get_tensor_info(tensor_id)
    print(f"\nExample tensor {tensor_id} info:")
    print(json.dumps(tensor_info, indent=2))
    
    # Example of getting shared memory group
    shared_group = parser.get_shared_memory_group(tensor_id)
    if shared_group:
        print(f"\nShared memory group for tensor {tensor_id}:")
        print(json.dumps(shared_group, indent=2))

if __name__ == "__main__":
    main()