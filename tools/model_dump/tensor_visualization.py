import plotly.graph_objs as go
import re
import os
import sys
from collections import defaultdict
import math
import json

class TensorAllocationVisualizer:
    def __init__(self):
        self.allocation_types = ['Arena RW', 'Mmap', 'Custom']
        self.color_map = {
            'Arena RW': 'rgb(31, 119, 180)',
            'Mmap': 'rgb(255, 127, 14)',
            'Custom': 'rgb(44, 160, 44)'
        }
        self.memory_spaces = defaultdict(list)
        self.shared_tensor_groups = []

    def format_bytes(self, size):
        """Format bytes size to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:3.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def find_shared_tensors(self, tensors):
        """Find groups of tensors that share the same memory space"""
        self.memory_spaces.clear()
        self.shared_tensor_groups.clear()
        
        for tensor in tensors:
            address = tensor['address']
            if address != 0:
                self.memory_spaces[address].append(tensor['tensor_index'])
        
        for address, tensor_indices in self.memory_spaces.items():
            if len(tensor_indices) > 1:
                self.shared_tensor_groups.append(tensor_indices)
        
        return self.shared_tensor_groups

    def get_tensor_usage(self, tensor_id, tensor_usage):
        """Get individual usage count and nodes for a tensor"""
        if tensor_id in tensor_usage:
            return {
                'count': tensor_usage[tensor_id]['count'],
                'nodes': sorted(tensor_usage[tensor_id]['nodes'])
            }
        else:
            # Return default values if tensor usage not found
            return {
                'count': 0,
                'nodes': []
            }

    def generate_report_data(self, tensors, tensor_usage):
        """Generate structured report data with individual tensor usage"""
        shared_groups = self.find_shared_tensors(tensors)
        
        report_data = {
            'summary': {
                'total_tensors': len(tensors),
                'total_size': sum(t['size'] for t in tensors),
                'shared_memory_groups': len(shared_groups),
                'allocation_types': {}
            },
            'tensors_by_type': defaultdict(list),
            'shared_memory_groups': []
        }

        # Add shared memory groups information
        for group in shared_groups:
            group_info = {
                'tensor_ids': group,
                'address': next(t['address'] for t in tensors if t['tensor_index'] == group[0]),
                'address_hex': next(t['address_hex'] for t in tensors if t['tensor_index'] == group[0]),
                'tensors': []
            }
            for tensor_id in group:
                tensor = next(t for t in tensors if t['tensor_index'] == tensor_id)
                group_info['tensors'].append({
                    'tensor_id': tensor_id,
                    'size': tensor['size'],
                    'data_type': tensor['data_type'],
                    'type': tensor['type']
                })
            report_data['shared_memory_groups'].append(group_info)

        # Group tensors by allocation type
        grouped_tensors = defaultdict(list)
        for tensor in tensors:
            grouped_tensors[tensor['type']].append(tensor)

        # Generate summary for each allocation type
        for alloc_type in self.allocation_types:
            type_tensors = grouped_tensors[alloc_type]
            if not type_tensors:
                continue

            type_size = sum(t['size'] for t in type_tensors)
            percentage = (type_size / report_data['summary']['total_size'] * 100) if report_data['summary']['total_size'] > 0 else 0

            report_data['summary']['allocation_types'][alloc_type] = {
                'count': len(type_tensors),
                'total_size': type_size,
                'percentage': percentage
            }

            # Generate detailed tensor information
            for tensor in sorted(type_tensors, key=lambda x: x['address']):
                tensor_id = tensor['tensor_index']
                individual_usage = self.get_tensor_usage(tensor_id, tensor_usage)
                
                # Find shared memory partners
                shared_with = []
                for group in shared_groups:
                    if tensor_id in group:
                        shared_with = [tid for tid in group if tid != tensor_id]
                        break
                
                tensor_data = {
                    'tensor_id': tensor_id,
                    'address': tensor['address'],
                    'address_hex': tensor['address_hex'],
                    'size': tensor['size'],
                    'size_formatted': self.format_bytes(tensor['size']),
                    'data_type': tensor['data_type'],
                    'usage_count': individual_usage['count'],
                    'used_by_nodes': individual_usage['nodes'],
                    'shape': tensor.get('shape', ''),
                    'shared_with': shared_with
                }
                report_data['tensors_by_type'][alloc_type].append(tensor_data)

        return report_data

    def save_report(self, report_data, output_file):
        """Save report to a text file with shared memory information"""
        with open(output_file, 'w') as f:
            f.write("=== TENSOR ALLOCATION REPORT ===\n\n")
            
            # Write summary
            total_size = report_data['summary']['total_size']
            f.write(f"Total tensors: {report_data['summary']['total_tensors']}\n")
            f.write(f"Total size: {self.format_bytes(total_size)}\n")
            f.write(f"Shared memory groups: {report_data['summary']['shared_memory_groups']}\n\n")
            
            # Write shared memory groups
            if report_data['shared_memory_groups']:
                f.write("=== SHARED MEMORY GROUPS ===\n")
                for group in report_data['shared_memory_groups']:
                    f.write(f"\n   Address {group['address_hex']}:\n")
                    for tensor in group['tensors']:
                        f.write(f"      Tensor {tensor['tensor_id']}: {self.format_bytes(tensor['size'])} "
                               f"({tensor['data_type']}, {tensor['type']})\n")
                f.write("\n")
            
            # Write details for each allocation type
            for alloc_type in self.allocation_types:
                if alloc_type not in report_data['tensors_by_type']:
                    continue
                    
                type_info = report_data['summary']['allocation_types'][alloc_type]
                f.write(f"=== {alloc_type} ===\n")
                f.write(f"   Number of tensors: {type_info['count']}\n")
                f.write(f"   Total size: {self.format_bytes(type_info['total_size'])} ({type_info['percentage']:.1f}% of total)\n\n")
                
                f.write("   Detailed tensor list:\n")
                f.write(f"   {'Tensor ID':<10} {'Address':<18} {'Size':<12} {'Data Type':<10} "
                       f"{'Usage Count':<12} {'Shared With':<15} {'Used By Nodes'}\n")
                f.write("   " + "-" * 140 + "\n")
                
                for tensor in report_data['tensors_by_type'][alloc_type]:
                    shared_str = f"[{', '.join(map(str, tensor['shared_with']))}]" if tensor['shared_with'] else "None"
                    nodes_str = f"Nodes: {', '.join(map(str, tensor['used_by_nodes']))}" if tensor['used_by_nodes'] else "Unused"
                    f.write(f"   {tensor['tensor_id']:<10} "
                           f"{tensor['address_hex']:<18} "
                           f"{tensor['size_formatted']:<12} "
                           f"{tensor['data_type']:<10} "
                           f"{tensor['usage_count']:<12} "
                           f"{shared_str:<15} "
                           f"{nodes_str}\n")
                f.write("\n")

    def print_tensor_report(self, report_data):
        """Print tensor report to console with shared memory information"""
        print("\n=== TENSOR ALLOCATION REPORT ===\n")
        
        # Print summary
        total_size = report_data['summary']['total_size']
        print(f"Total tensors: {report_data['summary']['total_tensors']}")
        print(f"Total size: {self.format_bytes(total_size)}")
        print(f"Shared memory groups: {report_data['summary']['shared_memory_groups']}\n")
        
        # Print shared memory groups
        if report_data['shared_memory_groups']:
            print("=== SHARED MEMORY GROUPS ===")
            for group in report_data['shared_memory_groups']:
                print(f"\n   Address {group['address_hex']}:")
                for tensor in group['tensors']:
                    print(f"      Tensor {tensor['tensor_id']}: {self.format_bytes(tensor['size'])} "
                          f"({tensor['data_type']}, {tensor['type']})")
            print()
        
        # Print details for each allocation type
        for alloc_type in self.allocation_types:
            if alloc_type not in report_data['tensors_by_type']:
                continue
                
            type_info = report_data['summary']['allocation_types'][alloc_type]
            print(f"=== {alloc_type} ===")
            print(f"   Number of tensors: {type_info['count']}")
            print(f"   Total size: {self.format_bytes(type_info['total_size'])} ({type_info['percentage']:.1f}% of total)\n")
            
            print("   Detailed tensor list:")
            print(f"   {'Tensor ID':<10} {'Address':<18} {'Size':<12} {'Data Type':<10} "
                  f"{'Usage Count':<12} {'Shared With':<15} {'Used By Nodes'}")
            print("   " + "-" * 140)
            
            for tensor in report_data['tensors_by_type'][alloc_type]:
                shared_str = f"[{', '.join(map(str, tensor['shared_with']))}]" if tensor['shared_with'] else "None"
                nodes_str = f"Nodes: {', '.join(map(str, tensor['used_by_nodes']))}" if tensor['used_by_nodes'] else "Unused"
                print(f"   {tensor['tensor_id']:<10} "
                      f"{tensor['address_hex']:<18} "
                      f"{tensor['size_formatted']:<12} "
                      f"{tensor['data_type']:<10} "
                      f"{tensor['usage_count']:<12} "
                      f"{shared_str:<15} "
                      f"{nodes_str}")
            print()
            
    def parse_tensor_details(self, file_path):
            """Parse tensor details from the TFLite allocation text file."""
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")

            with open(file_path, 'r') as file:
                content = file.read()

            # Split content by the actual node pattern "     X: [XXXX] OPERATOR"
            lines = content.split('\n')
            tensors = []
            seen_addresses = set()
            tensor_usage = defaultdict(lambda: {'count': 0, 'nodes': set()})

            # Look for node pattern like "     0: [0000] RESHAPE"
            node_pattern = re.compile(r'\s*(\d+):\s*\[(\d+)\]\s*(\w+)')
            current_node_idx = None
            
            print(f"Debug: Processing {len(lines)} lines")

            for line_num, line in enumerate(lines):
                # Check if this line starts a new node
                node_match = node_pattern.match(line)
                if node_match:
                    current_node_idx = int(node_match.group(1))
                    continue
                
                # Skip lines that don't have current node context
                if current_node_idx is None:
                    continue

                # Extract tensor information from this line
                # Pattern matches: "      Input 0: 39 Data Address: 0x... Type: INT32 Allocation Type: Arena RW Bytes: 4 Shape: [1, 1]"
                tensor_match = re.search(
                    r'(?:Input|Output|Temporary)\s+\d+:\s+(\d+)\s+Data\s+Address:\s+(0x[0-9a-fA-F]+)\s+Type:\s+(\w+)\s+Allocation\s+Type:\s+([\w\s]+?)\s+Bytes:\s+(\d+)\s+Shape:\s*\[(.*?)\]',
                    line
                )

                if tensor_match:
                    try:
                        tensor_index = int(tensor_match.group(1))
                        address_hex = tensor_match.group(2)
                        data_type = tensor_match.group(3)
                        alloc_type = tensor_match.group(4).strip()
                        bytes_size = int(tensor_match.group(5))
                        shape = tensor_match.group(6)

                        # Update tensor usage
                        tensor_usage[tensor_index]['count'] += 1
                        tensor_usage[tensor_index]['nodes'].add(current_node_idx)

                        address = int(address_hex, 16)

                        # Skip zero addresses
                        if address == 0:
                            continue

                        # For shared memory spaces, we now keep all tensors
                        if address in seen_addresses:
                            # Find existing tensor with same address
                            existing_tensors = [t for t in tensors if t['address'] == address]
                            if existing_tensors:
                                # Check if this is actually a different tensor sharing the same space
                                if all(t['tensor_index'] != tensor_index for t in existing_tensors):
                                    # This is a new tensor sharing memory with existing ones
                                    pass  # Continue to add it
                                else:
                                    continue  # Skip if it's the same tensor appearing multiple times
                        else:
                            seen_addresses.add(address)

                        # Normalize allocation type
                        if 'Arena RW' in alloc_type:
                            normalized_type = 'Arena RW'
                        elif 'Mmap' in alloc_type:
                            normalized_type = 'Mmap'
                        elif 'Custom' in alloc_type:
                            normalized_type = 'Custom'
                        else:
                            print(f"Warning: Unknown allocation type: {alloc_type}")
                            continue

                        tensor = {
                            'tensor_index': tensor_index,
                            'address': address,
                            'address_hex': address_hex,
                            'size': bytes_size,
                            'data_type': data_type,
                            'type': normalized_type,
                            'shape': shape
                        }
                        
                        tensors.append(tensor)

                    except Exception as e:
                        print(f"Error parsing tensor at line {line_num}: {e}")
                        print(f"Line content: {line}")
                        continue

            print(f"Debug: Found {len(tensors)} unique tensors")
            print(f"Debug: Tensor usage entries: {len(tensor_usage)}")

            # Sort tensors by address
            tensors = sorted(tensors, key=lambda x: x['address'])
            
            # Convert node sets to sorted lists in tensor_usage
            for usage in tensor_usage.values():
                usage['nodes'] = sorted(usage['nodes'])
            
            # Generate report data
            report_data = self.generate_report_data(tensors, tensor_usage)
            
            # Print report to console
            self.print_tensor_report(report_data)
            
            return tensors, tensor_usage, report_data
            
    def parse_execution_plan(self, content):
        """Parse execution plan from TFLite output"""
        execution_plan = []
        
        # Try different splitting patterns
        # Pattern 1: Split by numbered lines like "     0: [0000] RESHAPE"
        node_pattern = re.compile(r'\s*(\d+):\s*\[(\d+)\]\s*(\w+)')
        lines = content.split('\n')
        
        current_node = None
        for line in lines:
            # Look for node pattern like "     0: [0000] RESHAPE"
            node_match = node_pattern.match(line)
            if node_match:
                # Save previous node if exists
                if current_node:
                    execution_plan.append(current_node)
                
                # Start new node
                absolute_idx = int(node_match.group(1))
                tflite_idx = int(node_match.group(2))
                operator = node_match.group(3)
                
                current_node = {
                    'node_idx': absolute_idx,
                    'tflite_idx': tflite_idx,
                    'operator': operator,
                    'inputs': [],
                    'outputs': [],
                    'temporaries': []
                }
                continue
            
            # If we have a current node, look for tensor information
            if current_node:
                # Extract input tensors
                input_matches = re.finditer(r'Input\s+\d+:\s+(\d+)\s+Data\s+Address', line)
                for match in input_matches:
                    current_node['inputs'].append(int(match.group(1)))
                    
                # Extract output tensors
                output_matches = re.finditer(r'Output\s+\d+:\s+(\d+)\s+Data\s+Address', line)
                for match in output_matches:
                    current_node['outputs'].append(int(match.group(1)))
                    
                # Extract temporary tensors
                temp_matches = re.finditer(r'Temporary\s+\d+:\s+(\d+)\s+Data\s+Address', line)
                for match in temp_matches:
                    current_node['temporaries'].append(int(match.group(1)))
        
        # Don't forget the last node
        if current_node:
            execution_plan.append(current_node)
        
        print(f"Debug: Parsed {len(execution_plan)} nodes from execution plan")
        if execution_plan:
            print(f"Debug: First node: {execution_plan[0]}")
        
        return execution_plan

    def save_execution_plan(self, execution_plan, report_file):
        """Save execution plan to text file"""
        with open(report_file, 'a') as f:  # Append to existing report
            f.write("\n\n=== EXECUTION PLAN ===\n\n")
            f.write(f"   Total nodes: {len(execution_plan)}\n\n")
            
            for node in execution_plan:
                f.write(f"   Node {node['node_idx']}:\n")
                f.write(f"      Operator: {node['operator']}\n")
                f.write(f"      Input tensors: {node['inputs']}\n")
                f.write(f"      Output tensors: {node['outputs']}\n")
                if node['temporaries']:
                    f.write(f"      Temporary tensors: {node['temporaries']}\n")
                f.write("\n")

    def process_file(self, input_file, report_file=None, json_file=None):
        """
        Process the input file and generate reports
        
        Args:
            input_file: Path to input TFLite allocation file
            report_file: Optional path to save text report
            json_file: Optional path to save JSON data
        
        Returns:
            tuple: (tensors, tensor_usage, report_data, execution_plan)
        """
        with open(input_file, 'r') as f:
            content = f.read()
            
        # Parse tensor information
        tensors, tensor_usage, report_data = self.parse_tensor_details(input_file)
        
        # Parse execution plan
        execution_plan = self.parse_execution_plan(content)
        
        # Save text report if requested
        if report_file:
            self.save_report(report_data, report_file)
            self.save_execution_plan(execution_plan, report_file)
            print(f"\nReport saved to: {report_file}")
            
        # Save JSON data if requested
        if json_file:
            # Convert sets to lists for JSON serialization
            json_data = {
                'report_data': report_data,
                'tensor_usage': {
                    k: {'count': v['count'], 'nodes': sorted(v['nodes'])}
                    for k, v in tensor_usage.items()
                },
                'execution_plan': execution_plan
            }
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"JSON data saved to: {json_file}")
            
        return tensors, tensor_usage, report_data, execution_plan

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [report_file] [json_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    report_file = sys.argv[2] if len(sys.argv) > 2 else None
    json_file = sys.argv[3] if len(sys.argv) > 3 else None

    visualizer = TensorAllocationVisualizer()
    tensors, tensor_usage, report_data, execution_plan = visualizer.process_file(input_file, report_file, json_file)
    
    # Print summary of execution plan
    print(f"\nExecution Plan Summary:")
    print(f"Total nodes: {len(execution_plan)}")
    if execution_plan:
        print(f"First node: {execution_plan[0]}")
        print(f"Last node: {execution_plan[-1]}")
    else:
        print("No nodes found in execution plan. Check input file format.")

if __name__ == '__main__':
    main()