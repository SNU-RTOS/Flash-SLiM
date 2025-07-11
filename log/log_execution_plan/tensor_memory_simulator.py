# tensor_memory_simulator.py

import json
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from math import ceil

@dataclass
class Block:
    """Represents a memory block"""
    start_address: int
    size: int
    tensor_ids: Set[int]  # Set of tensor IDs sharing this block
    last_access: int
    dirty: bool = False

@dataclass
class MemoryEvent:
    """Represents a memory operation event"""
    step: int
    node_idx: int
    event_type: str  # 'load_block', 'evict_block', 'access_block'
    tensor_id: int
    block_address: int
    block_size: int
    shared_tensors: Set[int]  # Other tensors sharing this block
    is_write: bool = False

class MemorySimulator:
    def __init__(self, ram_size: int, tensor_data: Dict, execution_plan: List[Dict], 
                 block_size: int = 4096):  # Default 4KB block size
        """
        Initialize memory simulator with block-based paging and shared memory awareness
        
        Args:
            ram_size: Size of RAM in bytes
            tensor_data: Dictionary of tensor information
            execution_plan: List of nodes to execute
            block_size: Size of each memory block in bytes
        """
        self.ram_size = ram_size
        self.tensor_data = tensor_data
        self.execution_plan = execution_plan
        self.block_size = block_size
        
        # Calculate number of available blocks
        self.total_blocks = ram_size // block_size
        
        # Initialize statistics
        self.stats = {
            'block_hits': 0,
            'block_misses': 0,
            'block_evictions': 0,
            'total_io': 0,
            'peak_memory': 0,
            'tensor_hits': 0,
            'tensor_misses': 0,
            'shared_block_accesses': 0,
            'memory_saved_sharing': 0
        }
        
        # Runtime state
        self.ram_blocks = OrderedDict()  # Currently loaded blocks
        self.tensor_to_blocks = defaultdict(set)  # Maps tensor_id to its blocks
        self.current_ram_usage = 0
        
        # Event tracking
        self.memory_events = []
        
        # Build address mapping
        self.address_to_tensors = defaultdict(set)  # Maps memory address to tensor IDs
        self._build_address_mapping()

    def _get_block_boundaries(self, address: int) -> Tuple[int, int]:
        """Calculate block-aligned start and end addresses for a given address"""
        block_start = (address // self.block_size) * self.block_size
        block_end = block_start + self.block_size
        return block_start, block_end

    def _build_address_mapping(self) -> None:
        """Build mapping of addresses to tensors that share them, block-aligned"""
        # First, map raw addresses
        for tensor_id, tensor in self.tensor_data.items():
            block_start, _ = self._get_block_boundaries(tensor['address'])
            self.address_to_tensors[block_start].add(tensor_id)
            
        # Calculate memory saved through sharing
        total_tensor_size = sum(t['size'] for t in self.tensor_data.values())
        
        # Calculate unique block size needed
        unique_blocks = set()
        for tensor in self.tensor_data.values():
            start_addr = tensor['address']
            end_addr = start_addr + tensor['size']
            
            # Add all blocks this tensor spans
            start_block, _ = self._get_block_boundaries(start_addr)
            end_block, _ = self._get_block_boundaries(end_addr - 1)
            
            for block_addr in range(start_block, end_block + self.block_size, self.block_size):
                unique_blocks.add(block_addr)
        
        unique_memory_size = len(unique_blocks) * self.block_size
        self.stats['memory_saved_sharing'] = total_tensor_size - unique_memory_size

    def _find_tensors_in_block(self, block_start: int, block_end: int) -> Set[int]:
        """Find all tensors that have data in the given block range"""
        tensors_in_block = set()
        for tensor_id, tensor in self.tensor_data.items():
            tensor_start = tensor['address']
            tensor_end = tensor_start + tensor['size']
            
            # Check if tensor overlaps with block
            if not (tensor_end <= block_start or tensor_start >= block_end):
                tensors_in_block.add(tensor_id)
        
        return tensors_in_block

    def _calculate_blocks_for_tensor(self, tensor_id: int) -> List[Block]:
        """Calculate block-aligned blocks for a tensor"""
        tensor = self.tensor_data[tensor_id]
        tensor_start = tensor['address']
        tensor_end = tensor_start + tensor['size']
        
        # Calculate block-aligned boundaries
        start_block_addr, _ = self._get_block_boundaries(tensor_start)
        end_block_addr, _ = self._get_block_boundaries(tensor_end - 1)
        
        blocks = []
        for block_start in range(start_block_addr, end_block_addr + 1, self.block_size):
            block_end = block_start + self.block_size
            
            # Find all tensors that share this block
            tensor_ids = self._find_tensors_in_block(block_start, block_end)
            
            blocks.append(Block(
                start_address=block_start,
                size=self.block_size,  # Always use full block size
                tensor_ids=tensor_ids,
                last_access=0
            ))
        
        return blocks

    def _log_memory_event(self, step: int, node_idx: int, event_type: str, 
                         tensor_id: int, block_address: int, block_size: int,
                         shared_tensors: Set[int], is_write: bool = False) -> None:
        """Log a memory operation event"""
        event = MemoryEvent(
            step=step,
            node_idx=node_idx,
            event_type=event_type,
            tensor_id=tensor_id,
            block_address=block_address,
            block_size=block_size,
            shared_tensors=shared_tensors,
            is_write=is_write
        )
        self.memory_events.append(event)

    def _load_block(self, block: Block, step: int, node_idx: int) -> None:
        """Load a block into RAM, handling all tensors in the block"""
        if len(self.ram_blocks) >= self.total_blocks:
            self._evict_block(step, node_idx)
            
        block.last_access = step
        self.ram_blocks[block.start_address] = block
        
        # Update tensor_to_blocks for all tensors in this block
        for tid in block.tensor_ids:
            self.tensor_to_blocks[tid].add(block.start_address)
        
        self.current_ram_usage += block.size  # Always full block size
        self.stats['total_io'] += block.size
        self.stats['block_misses'] += 1
        
        # Log the loading of this block with all tensors it contains
        primary_tensor = min(block.tensor_ids)  # Use lowest tensor_id as primary
        other_tensors = block.tensor_ids - {primary_tensor}
        
        self._log_memory_event(
            step, node_idx, 'load_block',
            primary_tensor, block.start_address, block.size,
            other_tensors
        )
        
        # Print detailed block loading information
        tensor_details = []
        for tid in block.tensor_ids:
            tensor = self.tensor_data[tid]
            offset = tensor['address'] - block.start_address
            tensor_details.append(
                f"Tensor {tid} (offset: {offset}, size: {tensor['size']} bytes)"
            )
        
        print(f"  Loading block at 0x{block.start_address:x} ({block.size} bytes) containing:")
        for detail in tensor_details:
            print(f"    {detail}")

    def _evict_block(self, step: int, node_idx: int) -> None:
        """Evict a block using LRU policy, handling all tensors in the block"""
        if not self.ram_blocks:
            return
            
        # Find LRU block
        lru_addr, lru_block = next(iter(self.ram_blocks.items()))
        
        # Update statistics
        if lru_block.dirty:
            self.stats['total_io'] += self.block_size  # Always full block size
            
        # Remove block from all tensors that were using it
        for tid in lru_block.tensor_ids:
            if lru_addr in self.tensor_to_blocks[tid]:
                self.tensor_to_blocks[tid].remove(lru_addr)
            
        self.ram_blocks.pop(lru_addr)
        self.current_ram_usage -= self.block_size
        self.stats['block_evictions'] += 1
        
        # Log eviction with all affected tensors
        primary_tensor = min(lru_block.tensor_ids)
        other_tensors = lru_block.tensor_ids - {primary_tensor}
        self._log_memory_event(
            step, node_idx, 'evict_block',
            primary_tensor, lru_addr, self.block_size,
            other_tensors
        )

    def _access_tensor(self, tensor_id: int, step: int, node_idx: int, 
                      is_write: bool = False) -> None:
        """Access a tensor, loading necessary blocks"""
        if tensor_id not in self.tensor_data:
            return
            
        blocks = self._calculate_blocks_for_tensor(tensor_id)
        tensor_fully_loaded = True
        
        for block in blocks:
            if block.start_address in self.ram_blocks:
                # Update access time and move to end of LRU order
                existing_block = self.ram_blocks[block.start_address]
                existing_block.last_access = step
                if is_write:
                    existing_block.dirty = True
                self.ram_blocks.move_to_end(block.start_address)
                self.stats['block_hits'] += 1
                
                # Log access to this block and all tensors it contains
                self._log_memory_event(
                    step, node_idx, 'access_block',
                    tensor_id, block.start_address, block.size,
                    existing_block.tensor_ids - {tensor_id},
                    is_write
                )
            else:
                self._load_block(block, step, node_idx)
                tensor_fully_loaded = False
        
        # Update tensor-level statistics
        if tensor_fully_loaded:
            self.stats['tensor_hits'] += 1
        else:
            self.stats['tensor_misses'] += 1

    def simulate(self) -> Dict:
        """Run memory simulation with block-aligned access"""
        print(f"\nStarting block-based memory simulation with shared memory awareness...")
        print(f"RAM size: {self.ram_size} bytes, Block size: {self.block_size} bytes")
        print(f"Total blocks available: {self.total_blocks}")
        
        # Print initial block alignment information
        print("\nBlock alignment analysis:")
        misaligned_tensors = []
        for tensor_id, tensor in self.tensor_data.items():
            if tensor['address'] % self.block_size != 0:
                misaligned_tensors.append(tensor_id)
        if misaligned_tensors:
            print(f"Warning: Found {len(misaligned_tensors)} tensors not aligned to {self.block_size}-byte boundaries")
            print(f"Misaligned tensors: {misaligned_tensors}")
        
        for step, node in enumerate(self.execution_plan):
            node_idx = node['node_idx']
            print(f"\nStep {step}: Processing node {node_idx} ({node['operator']})")
            
            # Process input tensors (read)
            for tensor_id in node['inputs']:
                if tensor_id in self.tensor_data:
                    tensor = self.tensor_data[tensor_id]
                    start_block, _ = self._get_block_boundaries(tensor['address'])
                    tensors_in_block = self._find_tensors_in_block(
                        start_block, start_block + self.block_size
                    )
                    shared_str = f" (shares block with tensors {tensors_in_block - {tensor_id}})" if len(tensors_in_block) > 1 else ""
                    
                    print(f"  Reading tensor {tensor_id} "
                          f"(size: {tensor['size']} bytes, "
                          f"block aligned: {tensor['address'] % self.block_size == 0})"
                          f"{shared_str}")
                    self._access_tensor(tensor_id, step, node_idx, is_write=False)
            
            # Process output tensors (write)
            for tensor_id in node['outputs']:
                if tensor_id in self.tensor_data:
                    tensor = self.tensor_data[tensor_id]
                    start_block, _ = self._get_block_boundaries(tensor['address'])
                    tensors_in_block = self._find_tensors_in_block(
                        start_block, start_block + self.block_size
                    )
                    shared_str = f" (shares block with tensors {tensors_in_block - {tensor_id}})" if len(tensors_in_block) > 1 else ""
                    
                    print(f"  Writing tensor {tensor_id} "
                          f"(size: {tensor['size']} bytes, "
                          f"block aligned: {tensor['address'] % self.block_size == 0})"
                          f"{shared_str}")
                    self._access_tensor(tensor_id, step, node_idx, is_write=True)
            
            # Update peak memory usage
            self.stats['peak_memory'] = max(self.stats['peak_memory'], 
                                          self.current_ram_usage)
            
            # self._print_memory_state()
        
        return self.stats

    def _print_memory_state(self) -> None:
        """Print current state of memory with block information"""
        print(f"\nCurrent Memory State:")
        print(f"  RAM Usage: {self.current_ram_usage}/{self.ram_size} bytes "
              f"({self.current_ram_usage/self.ram_size*100:.1f}%)")
        print(f"  Loaded Blocks: {len(self.ram_blocks)}/{self.total_blocks}")
        
        if self.ram_blocks:
            print(f"  Average tensors per block: "
                  f"{sum(len(b.tensor_ids) for b in self.ram_blocks.values())/len(self.ram_blocks):.2f}")
        
        # Print shared block information
        shared_blocks = [(addr, block) for addr, block in self.ram_blocks.items() 
                        if len(block.tensor_ids) > 1]
        if shared_blocks:
            print("\n  Shared Blocks:")
            for addr, block in shared_blocks[:5]:  # Show first 5 shared blocks
                print(f"    Block 0x{addr:x}: {len(block.tensor_ids)} tensors "
                      f"{sorted(block.tensor_ids)}")
            if len(shared_blocks) > 5:
                print(f"    ... and {len(shared_blocks)-5} more shared blocks")

    def get_hit_ratios(self) -> Dict[str, float]:
        """Calculate hit ratios at block and tensor levels"""
        block_accesses = self.stats['block_hits'] + self.stats['block_misses']
        tensor_accesses = self.stats['tensor_hits'] + self.stats['tensor_misses']
        total_accesses = max(1, self.stats['block_hits'] + self.stats['block_misses'])
        
        return {
            'block_hit_ratio': self.stats['block_hits'] / block_accesses if block_accesses > 0 else 0,
            'tensor_hit_ratio': self.stats['tensor_hits'] / tensor_accesses if tensor_accesses > 0 else 0,
            'shared_access_ratio': self.stats['shared_block_accesses'] / total_accesses
        }

    def print_memory_events(self) -> None:
        """Print chronological list of memory events"""
        print("\n=== Memory Event Log ===")
        current_step = -1
        
        for event in self.memory_events:
            if event.step != current_step:
                current_step = event.step
                print(f"\nStep {current_step}:")
            
            shared_str = ""
            if event.shared_tensors:
                shared_str = f" (shared with tensors {sorted(event.shared_tensors)})"
            
            event_str = (f"  Node {event.node_idx}: {event.event_type} - "
                        f"Tensor {event.tensor_id}{shared_str}")
            if event.event_type == 'access_block':
                event_str += f" ({'write' if event.is_write else 'read'})"
            event_str += f" [Block addr: 0x{event.block_address:x}, size: {event.block_size}]"
            print(event_str)

    def print_report(self) -> None:
        """Print detailed simulation results with block alignment information"""
        hit_ratios = self.get_hit_ratios()
        
        print("\n=== Memory Simulation Report ===")
        print(f"Configuration:")
        print(f"  RAM Size: {self.ram_size} bytes")
        print(f"  Block Size: {self.block_size} bytes")
        print(f"  Total Blocks: {self.total_blocks}")
        
        print("\nBlock Alignment Analysis:")
        aligned_count = sum(1 for t in self.tensor_data.values() 
                          if t['address'] % self.block_size == 0)
        print(f"  Block-aligned tensors: {aligned_count}/{len(self.tensor_data)}")
        
        print("\nShared Memory Statistics:")
        print(f"  Memory Saved Through Sharing: {self.stats['memory_saved_sharing']} bytes")
        print(f"  Shared Block Access Ratio: {hit_ratios['shared_access_ratio']:.4f}")
        print(f"  Total Unique Blocks Used: {len(set(b.start_address for b in self.ram_blocks.values()))}")
        
        print("\nPerformance Metrics:")
        print(f"  Block-level Hit Ratio: {hit_ratios['block_hit_ratio']:.4f}")
        print(f"  Tensor-level Hit Ratio: {hit_ratios['tensor_hit_ratio']:.4f}")
        print(f"  Peak Memory Usage: {self.stats['peak_memory']} bytes")
        print(f"  Total I/O: {self.stats['total_io']} bytes")
        
        print("\nDetailed Statistics:")
        print(f"  Block Hits: {self.stats['block_hits']}")
        print(f"  Block Misses: {self.stats['block_misses']}")
        print(f"  Block Evictions: {self.stats['block_evictions']}")
        print(f"  Tensor Hits: {self.stats['tensor_hits']}")
        print(f"  Tensor Misses: {self.stats['tensor_misses']}")
        print(f"  Shared Block Accesses: {self.stats['shared_block_accesses']}")
        
        # Print block sharing analysis
        print("\nBlock Sharing Analysis:")
        block_sharing_stats = defaultdict(int)
        for block in self.ram_blocks.values():
            block_sharing_stats[len(block.tensor_ids)] += 1
        
        for tensors_count, blocks_count in sorted(block_sharing_stats.items()):
            print(f"  Blocks with {tensors_count} tensors: {blocks_count}")
        
        # Print memory sharing groups
        print("\nShared Memory Groups (by block):")
        shared_blocks = [(addr, block) for addr, block in self.ram_blocks.items() 
                        if len(block.tensor_ids) > 1]
        for addr, block in shared_blocks:
            print(f"\n  Block at 0x{addr:x}:")
            for tid in sorted(block.tensor_ids):
                tensor = self.tensor_data[tid]
                offset = tensor['address'] - addr
                print(f"    Tensor {tid}: offset {offset} bytes, size {tensor['size']} bytes")
        
        # Print chronological event log
        self.print_memory_events()


def main():
    """Example usage"""
    import sys
    from tensor_data_parser import TensorDataParser
    
    if len(sys.argv) < 2:
        print("Usage: python memory_simulator.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Parse tensor data
    parser = TensorDataParser(json_file)
    parser.load_data()
    
    # Example configuration
    ram_size = 4 * 1024 * 1024 * 1024  # 4 GB
    block_size = 4096  # 4 KB blocks
    
    # Run simulation
    simulator = MemorySimulator(ram_size, parser.tensors, parser.execution_plan, 
                              block_size=block_size)
    stats = simulator.simulate()
    simulator.print_report()

if __name__ == "__main__":
    main()