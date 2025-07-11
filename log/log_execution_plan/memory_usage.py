import re
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

def parse_execution_plan(file_path):
    # [Previous parsing code remains the same as in original]
    with open(file_path, 'r') as file:
        content = file.read()

    nodes = []
    current_node = None
    current_tensor_type = None
    tensor_access_count = defaultdict(int)
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        node_match = re.match(r'Node (\d+):', line)
        if node_match:
            if current_node:
                nodes.append(current_node)
            current_node = {
                'node_id': int(node_match.group(1)),
                'operator': None,
                'input_tensors': [],
                'output_tensors': [],
                'intermediate_tensors': [],
                'temporary_tensors': [],
                'total_memory': 0,
                'tensor_ids': []
            }
            continue
            
        # [Rest of the parsing logic remains the same]
        if line.startswith('Operator:'):
            if current_node:
                current_node['operator'] = line.split('Operator:')[1].strip()
            continue
            
        if line in ['Input Tensors:', 'Output Tensors:', 'Intermediate Tensors:', 'Temporary Tensors:']:
            current_tensor_type = line
            continue
            
        tensor_info_match = re.search(r'(?:Input|Output|Intermediate|Temporary) \d+: (\d+)', line)
        if tensor_info_match and current_node:
            tensor_id = int(tensor_info_match.group(1))
            current_node['tensor_ids'].append(tensor_id)
            tensor_access_count[tensor_id] += 1

        if 'Bytes:' in line:
            bytes_match = re.search(r'Bytes: (\d+)', line)
            if bytes_match and current_node and current_tensor_type:
                bytes_used = int(bytes_match.group(1))
                
                if current_tensor_type == 'Input Tensors:':
                    current_node['input_tensors'].append(bytes_used)
                elif current_tensor_type == 'Output Tensors:':
                    current_node['output_tensors'].append(bytes_used)
                elif current_tensor_type == 'Intermediate Tensors:':
                    current_node['intermediate_tensors'].append(bytes_used)
                elif current_tensor_type == 'Temporary Tensors:':
                    current_node['temporary_tensors'].append(bytes_used)
    
    if current_node:
        nodes.append(current_node)
    
    for node in nodes:
        node['total_memory'] = (
            sum(node['input_tensors']) +
            sum(node['output_tensors']) +
            sum(node['intermediate_tensors']) +
            sum(node['temporary_tensors'])
        )
        
        node['input_memory'] = sum(node['input_tensors'])
        node['output_memory'] = sum(node['output_tensors'])
        node['intermediate_memory'] = sum(node['intermediate_tensors'])
        node['temporary_memory'] = sum(node['temporary_tensors'])
    
    return nodes, tensor_access_count

def create_individual_plots(nodes, tensor_access_count):
    # Sort nodes by execution order
    nodes.sort(key=lambda x: x['node_id'])
    
    # Extract data
    node_ids = [node['node_id'] for node in nodes]
    node_operators = [node['operator'] for node in nodes]
    
    # Convert memory to KB
    total_memories = [node['total_memory'] / 1024 for node in nodes]
    input_memories = [node['input_memory'] / 1024 for node in nodes]
    output_memories = [node['output_memory'] / 1024 for node in nodes]
    intermediate_memories = [node['intermediate_memory'] / 1024 for node in nodes]
    temporary_memories = [node['temporary_memory'] / 1024 for node in nodes]
    
    # Common hover template
    hover_template = (
        "<b>Node %{x}</b><br>" +
        "Operator: %{customdata}<br>" +
        "Memory: %{y:.2f} KB<br>" +
        "<extra></extra>"
    )
    
    # Color scheme
    colors = {
        'total': 'rgba(49, 130, 189, 0.8)',
        'input': 'rgba(55, 83, 109, 0.8)',
        'output': 'rgba(26, 118, 255, 0.8)',
        'intermediate': 'rgba(78, 186, 111, 0.8)',
        'temporary': 'rgba(222, 84, 84, 0.8)',
        'access': 'rgba(255, 87, 51, 0.8)'
    }
    
    # Create individual plots with improved bar width and x-axis visibility
    plots = []
    
    # Memory usage plots
    memory_data = [
        (total_memories, 'Total Memory Usage by Node', colors['total']),
        (input_memories, 'Input Memory Usage by Node', colors['input']),
        (output_memories, 'Output Memory Usage by Node', colors['output']),
        (intermediate_memories, 'Intermediate Memory Usage by Node', colors['intermediate']),
        (temporary_memories, 'Temporary Memory Usage by Node', colors['temporary'])
    ]
    
    for memories, title, color in memory_data:
        # Calculate non-overlapping bar width
        # Ensure bars are separated by using a width less than 1
        # This guarantees space between bars since node_ids are integers
        bar_width = 0.8  # Width less than 1 ensures no overlap
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=title,
                x=node_ids,
                y=memories,
                customdata=node_operators,
                marker=dict(
                    color=color,
                    line=dict(color=color.replace('0.8', '1'), width=1)
                ),
                width=bar_width,
                hovertemplate=hover_template
            )
        )
        
        # Update layout for better visibility
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            plot_bgcolor='white',
            height=600,
            width=1800,
            xaxis=dict(
                title="Node Execution Order",
                showgrid=True,
                gridcolor='lightgrey',
                tickmode='auto',
                nticks=30,  # Increased number of x-axis ticks
                range=[min(node_ids) - 1, max(node_ids) + 1],  # Add padding to x-axis range
                showticklabels=True
            ),
            yaxis=dict(
                title="Memory Usage (KB)",
                type="log",
                showgrid=True,
                gridcolor='lightgrey'
            )
        )
        plots.append(fig)
    
    # Tensor access frequency plot
    tensor_ids = list(tensor_access_count.keys())
    access_counts = list(tensor_access_count.values())
    
    # Set non-overlapping bar width for tensor access plot
    tensor_bar_width = 0.8  # Width less than 1 ensures no overlap
    
    access_fig = go.Figure()
    access_fig.add_trace(
        go.Bar(
            name='Access Frequency',
            x=tensor_ids,
            y=access_counts,
            marker=dict(
                color=colors['access'],
                line=dict(color=colors['access'].replace('0.8', '1'), width=1)
            ),
            width=tensor_bar_width,
            hovertemplate="<b>Tensor %{x}</b><br>Accesses: %{y}<br><extra></extra>"
        )
    )
    
    access_fig.update_layout(
        title=dict(
            text="Tensor Access Frequency",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=False,
        plot_bgcolor='white',
        height=900,
        width=1600,
        xaxis=dict(
            title="Tensor ID",
            showgrid=True,
            gridcolor='lightgrey',
            tickmode='auto',
            nticks=30,
            range=[min(tensor_ids) - 1, max(tensor_ids) + 1],  # Add padding to x-axis range
            showticklabels=True
        ),
        yaxis=dict(
            title="Number of Accesses",
            showgrid=True,
            gridcolor='lightgrey'
        )
    )
    plots.append(access_fig)
    
    # Print statistics
    print("\nMemory Usage Statistics:")
    print(f"Average Memory per Node: {sum(total_memories) / len(total_memories):.2f} KB")
    print(f"Maximum Memory Usage: {max(total_memories):.2f} KB")
    print(f"Total Memory Usage: {sum(total_memories):.2f} KB")
    
    print("\nTensor Access Frequency Analysis:")
    print(f"Total number of unique tensors: {len(tensor_access_count)}")
    print(f"Maximum access frequency: {max(access_counts)}")
    print(f"Average access frequency: {sum(access_counts)/len(access_counts):.2f}")
    
    print("\nMost frequently accessed tensors (top 10):")
    sorted_tensors = sorted(tensor_access_count.items(), key=lambda x: x[1], reverse=True)[:10]
    for tensor_id, count in sorted_tensors:
        print(f"Tensor {tensor_id}: {count} accesses")
    
    return plots

def main():
    # Parse the execution plan
    nodes, tensor_access_count = parse_execution_plan('XNNPACK_X.txt')
    
    # Create individual plots
    plots = create_individual_plots(nodes, tensor_access_count)
    
    # Show each plot
    for plot in plots:
        plot.show()

if __name__ == "__main__":
    main()