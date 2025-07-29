import os
import re
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhaseMetrics:
    def __init__(self):
        self.duration = 0
        self.minor_faults = 0
        self.major_faults = 0

class LogMetrics:
    def __init__(self):
        self.phases: Dict[str, PhaseMetrics] = {}
        self.decode_times: List[float] = []
        self.prefill_time = 0
        self.time_to_first_token = 0
        self.is_valid = True  # Flag to track if all required metrics were found
        self.decoding_time_breakdown: Dict[str, float] = {
            'avg_latency': 0.0,
            'user_time': 0.0,
            'system_time': 0.0
        }
        # Added for detailed TTFT breakdown
        self.prefill_time_breakdown: Dict[str, float] = {
            'cpu_time': 0.0,
            'user_time': 0.0,
            'system_time': 0.0
        }
        self.first_decode_breakdown: Dict[str, float] = {
            'cpu_time': 0.0,
            'user_time': 0.0,
            'system_time': 0.0
        }
        
    def calculate_total_major_faults(self) -> int:
        return sum(phase.major_faults for phase in self.phases.values())
    


############################################
# Utility Functions
############################################
def __convert_str2float_safe(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with error handling."""
    try:
        return float(value.replace(',', ''))
    except (ValueError, AttributeError):
        logger.warning(f"Could not convert '{value}' to float, using default: {default}")
        return default
    
def __get_ram_size_info_from_dir(dir_name: str) -> Optional[str]:
    """Get RAM size from directory name in the format result_dp_[]G or result_dp_[]M"""
    match = re.match(r'result_dp_(\d+)(G|M)', dir_name)
    if match:
        size = int(match.group(1))
        unit = match.group(2)
        if unit == 'M':
            return f"{size}M"
        else:  # unit == 'G'
            return f"{size}G"
    return None

def get_available_ram_sizes(base_dir: str) -> List[str]:
    """Get list of available RAM sizes from directory names"""
    try:
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        ram_sizes = []
        for dir_name in dirs:
            ram_size = __get_ram_size_info_from_dir(dir_name)
            if ram_size:
                ram_sizes.append(ram_size)
        return sorted(ram_sizes, key=lambda x: (x[-1], int(x[:-1])))  # Sort by unit (M then G) and then by size
    except Exception as e:
        logger.error(f"Error getting RAM sizes: {e}")
        return []

def get_ram_dir_name(ram_size: str) -> str:
    """Convert RAM size (e.g., '4G' or '512M') to directory name format"""
    return f"result_dp_{ram_size}"

def export_results_to_csv(results, output_file='results.csv'):
    '''
    Export the results to a CSV file with the following columns:
    '''

    sample_ram = next(iter(results))
    sample_token = next(iter(results[sample_ram]))
    sample_metric: LogMetrics = results[sample_ram][sample_token][0]
    
    # static_fields = [
    #     'prefill_time [ms]', 'time_to_first_token [ms]',
    # ]

    decoding_keys = list(sample_metric.decoding_time_breakdown.keys())
    prefill_keys = list(sample_metric.prefill_time_breakdown.keys())
    first_decode_keys = list(sample_metric.first_decode_breakdown.keys())
    phase_keys = [phase_name for phase_name in list(sample_metric.phases.keys()) if phase_name != "Decode"]

    decoding_fields = [f'decoding_{k} [sec]' for k in decoding_keys]
    prefill_fields = [f'prefill_{k} [sec]' for k in prefill_keys]
    first_decode_fields = [f'first_decode_{k} [sec]' for k in first_decode_keys ]
    metric_phases_fields = [f'{k} [ms]' for k in reversed(phase_keys)]
    
    headers = ['ram', 'input_tokens']  + decoding_fields + prefill_fields + first_decode_fields + metric_phases_fields

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for ram_size, token_dict in results.items():
            for input_tokens, metrics_list in token_dict.items():
                for metric in metrics_list:
                    row = [
                        ram_size,
                        input_tokens,
                        # metric.prefill_time,
                        # metric.time_to_first_token,
                    ]

                    row += [metric.decoding_time_breakdown[k] for k in decoding_keys]
                    row += [metric.prefill_time_breakdown[k] for k in prefill_keys]
                    row += [metric.first_decode_breakdown[k] for k in first_decode_keys]
                    row += [metric.phases[k].duration for k in reversed(phase_keys)]
                    
                    writer.writerow(row)
                    

    
    logger.info(f"Exported results to {output_file}")


############################################
# Parsing Functions
############################################

## from Perf or Rusage
def __parse_phase_info(content: str) -> Dict[str, PhaseMetrics]:
    phases = {}
    
    # First try to parse using the new detailed performance statistics format
    new_format_pattern = r"=== Performance Statistics for Phase: ([^\n=]+) ===\s*\n(?:Number of measurements: \d+\s*\n)?(?:Average )?Wall clock time: (\d+) ms\s*\n(?:Average )?User time: ([\d.e-]+) sec\s*\n(?:Average )?System time: ([\d.e-]+) sec"
    new_format_matches = list(re.finditer(new_format_pattern, content))
    
    # Flag to track if we found the Decode_Token phase
    found_decode_token = False
    
    # If we found the new format
    if new_format_matches:
        logger.info(f"Found new format performance statistics for {len(new_format_matches)} phases")
        
        for match in new_format_matches:
            try:
                phase_name = match.group(1).strip()
                metrics = PhaseMetrics()
                metrics.duration = int(match.group(2))
                
                # For page faults, we don't have direct numbers in the new format
                # We'll try to estimate or use zero as placeholder
                metrics.minor_faults = 0
                metrics.major_faults = 0
                
                # Special case: rename "Decode_Token" to "Decode" for consistency
                if phase_name == "Decode_Token":
                    phase_name = "Decode"
                    found_decode_token = True
                    logger.info("Renamed Decode_Token phase to Decode for consistency")
                
                # Add or update phase metrics
                phases[phase_name] = metrics
                logger.debug(f"Parsed phase {phase_name} with new format: duration={metrics.duration}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing phase info with new format for {match.group(1)}: {e}")
    
    # If we haven't found Decode or Decode_Token in the new format, try the original methods
    if 'Decode' not in phases and not found_decode_token:
        # Try to parse decode step-level faults (original format)
        decode_step_pattern = r"Decode step (\d+)\s*\n\s*- Major page faults: (\d+)\s*\n\s*- Minor page faults: (\d+)"
        decode_step_matches = list(re.finditer(decode_step_pattern, content))
        
        if decode_step_matches:
            major_faults = [int(match.group(2)) for match in decode_step_matches]
            minor_faults = [int(match.group(3)) for match in decode_step_matches]
            
            decode_metrics = PhaseMetrics()
            decode_metrics.major_faults = np.mean(major_faults)
            decode_metrics.minor_faults = np.mean(minor_faults)
            phases['Decode'] = decode_metrics
            
            logger.info(f"Parsed Decode step faults: Avg Major={decode_metrics.major_faults:.2f}, Avg Minor={decode_metrics.minor_faults:.2f}")
        else:
            # Fallback to previous parsing method if step-level data not found
            logger.warning("No step-level decode faults found, falling back to summary parsing")
            decode_phase_pattern = r"Phase: Decode\s*\nNumber of measurements: (\d+)\s*\nAverage duration: ([\d.]+) ms\s*\nTotal minor page faults: (\d+)\s*\nTotal major page faults: (\d+)"
            decode_match = re.search(decode_phase_pattern, content)
            
            if decode_match:
                decode_metrics = PhaseMetrics()
                decode_metrics.duration = float(decode_match.group(2))
                decode_metrics.minor_faults = int(decode_match.group(3))
                decode_metrics.major_faults = int(decode_match.group(4))
                phases['Decode'] = decode_metrics
    
    # Parse other phases with the old format if they haven't been found yet
    phase_pattern = r"Phase: ([^\n]+)\s*\nDuration: (\d+) ms\s*\nMinor page faults: (\d+)\s*\nMajor page faults: (\d+)"
    matches = re.finditer(phase_pattern, content)
    
    for match in matches:
        try:
            phase_name = match.group(1).strip()
            if phase_name not in phases:  # Only add if not already parsed with new format
                metrics = PhaseMetrics()
                metrics.duration = int(match.group(2))
                metrics.minor_faults = int(match.group(3))
                metrics.major_faults = int(match.group(4))
                phases[phase_name] = metrics
                logger.debug(f"Parsed phase {phase_name} with old format: duration={metrics.duration}, minor={metrics.minor_faults}, major={metrics.major_faults}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing phase info for {phase_name}: {e}")
            continue
    # print(phases.keys())
    return phases

## from Perf or Rusage
def __parse_decode_time(content: str) -> List[float]:
    """Parse decode times from both old and new formats, extracting CPU time"""
    times = []
    
    # Try new format first (Per-step details)
    new_decode_pattern = r"Step (\d+):\s*\n  Wall clock time: \d+ ms\s*\n  User time: [\d.e-]+ sec\s*\n  System time: [\d.e-]+ sec\s*\n  Total CPU time \(user\+system\): ([\d.e-]+) sec"
    new_matches = list(re.finditer(new_decode_pattern, content))
    
    if new_matches:
        logger.info(f"Found new format decode time details for {len(new_matches)} steps")
        for match in new_matches:
            try:
                step = int(match.group(1))
                cpu_time = float(match.group(2))
                # Skip step 0 for consistency with old format
                if step > 0:
                    times.append(cpu_time)
                    logger.debug(f"Parsed decode time for step {step}: {cpu_time} seconds")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing decode time from new format: {e}")
    
    # If no times found with new format, try the old format
    if not times:
        old_decode_pattern = r"Decode \d+ took\s*\n-\s*([\d.]+)\s*\[sec\] CPU time"
        old_matches = list(re.finditer(old_decode_pattern, content))
        
        if old_matches:
            i = 0
            for match in old_matches:
                try:
                    cpu_time = float(match.group(1))
                    # Skip the first decode (index 0) as per original implementation
                    if i == 0:
                        i += 1
                        continue
                    times.append(cpu_time)
                    logger.debug(f"Parsed decode time with old format: {cpu_time} seconds")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing decode time from old format: {e}")
                    continue
    
    if times:
        logger.info(f"Successfully parsed {len(times)} decode times")
    else:
        logger.warning("No decode time entries found in either format")
    
    return times

## from Metric point and Rusage
def __parse_decode_time_breakdown(content: str) -> Dict[str, float]:
    """
    Parse decode time breakdown from log content.
    
    Returns a dictionary with the following keys:
    - 'avg_latency': Average Decoding Latency (in seconds)
    - 'user_time': User time
    - 'system_time': System time
    """
    breakdown = {
        'avg_latency': 0.0,
        'user_time': 0.0,
        'system_time': 0.0
    }
    
    # Parse average decoding latency - fixed pattern to match the actual format in logs
    avg_latency_pattern = r"\[METRICS\]\s*Average Decoding Latency\s*:\s*([\d.,]+)\s*ms/tokens"
    avg_latency_match = re.search(avg_latency_pattern, content)
    if avg_latency_match:
        # Convert from milliseconds to seconds for consistency with other time measurements
        ms_value = __convert_str2float_safe(avg_latency_match.group(1))
        breakdown['avg_latency'] = ms_value / 1000.0  # Convert ms to seconds
    
    # Parse time breakdown
    block_pattern = r"(Decode\s+(\d+)\s+took\s*\n(?:\s*-\s*[\d.]+\s*\[sec\]\s*\w+ time\s*\n?){2,3})"
    blocks = re.finditer(block_pattern, content)
    
    for block_match in blocks:
        block_text = block_match.group(1)  # 전체 Decode 블록
        phase_id = int(block_match.group(2))  # "Decode 9"의 숫자
        
        # 2. 블록 안에서 시간별 항목 추출
        time_pattern = r"-\s*([\d.]+)\s*\[sec\]\s+(CPU|User|System) time"
        for time_match in re.finditer(time_pattern, block_text):
            time_value = __convert_str2float_safe(time_match.group(1))
            time_type = time_match.group(2).lower()
            # print(time_value, time_type)
            if time_type == 'user':
                breakdown['user_time'] = time_value
            elif time_type == 'system':
                breakdown['system_time'] = time_value
        
    return breakdown

## from ScopeTimer, Metric point
def __parse_prefill_and_ttft(content: str) -> Tuple[float, float]:
    """Parse prefill time and time to first token with more flexible patterns"""
    prefill_pattern = r"\[INFO\]\s*Prefill took\s*([\d,]+(?:\.\d+)?)\s*ms"
    ttft_pattern = r"\[METRICS\]\s*Time To First Token\s*:\s*([\d,]+(?:\.\d+)?)\s*ms"
    
    prefill_match = re.search(prefill_pattern, content)
    ttft_match = re.search(ttft_pattern, content)
    
    prefill_time = __convert_str2float_safe(prefill_match.group(1)) if prefill_match else 0
    ttft = __convert_str2float_safe(ttft_match.group(1)) if ttft_match else 0
    
    if prefill_match and ttft_match:
        logger.info(f"Parsed prefill={prefill_time}ms, ttft={ttft}ms")
    else:
        logger.warning("Failed to parse prefill or TTFT values")
    
    return prefill_time, ttft

## from Perf or Rusage
def __parse_prefill_breakdown(content: str) -> Dict[str, float]:
    """
    Parse prefill time breakdown from both old and new log content formats.
    
    Returns a dictionary with the following keys:
    - 'cpu_time': CPU time for prefill
    - 'user_time': User time for prefill
    - 'system_time': System time for prefill
    """
    breakdown = {
        'cpu_time': 0.0,
        'user_time': 0.0,
        'system_time': 0.0
    }
    
    # Try new format first
    new_format_pattern = r"=== Performance Statistics for Phase: Prefill ===\s*\nWall clock time: \d+ ms\s*\nUser time: ([\d.e-]+) sec\s*\nSystem time: ([\d.e-]+) sec\s*\nTotal CPU time \(user\+system\): ([\d.e-]+) sec"
    new_match = re.search(new_format_pattern, content)
    
    if new_match:
        # Extract values from the new format
        breakdown['user_time'] = __convert_str2float_safe(new_match.group(1))
        breakdown['system_time'] = __convert_str2float_safe(new_match.group(2))
        breakdown['cpu_time'] = __convert_str2float_safe(new_match.group(3))
        logger.info(f"Parsed prefill breakdown from new format: user={breakdown['user_time']}, system={breakdown['system_time']}, cpu={breakdown['cpu_time']}")
        return breakdown
    
    # If new format not found, try old format
    # Find the line with "Prefill Stage took" to locate the section
    prefill_marker = "Prefill took"
    prefill_pos = content.find(prefill_marker)
    
    if prefill_pos != -1:
        # Look for time measurements following that section
        # Find all time measurements following the pattern "- X [sec] Type time"
        time_pattern = r"-\s*([\d.]+)\s*\[sec\] (CPU|User|System) time"
        
        # Extract a reasonable chunk of text after the marker
        section_end = content.find("\n\n", prefill_pos)
        if section_end == -1:  # If no double newline, take a reasonable amount
            section_end = prefill_pos + 500
        
        prefill_section = content[prefill_pos:section_end]
        
        # Find all time measurements in this section
        for match in re.finditer(time_pattern, prefill_section):
            time_value = __convert_str2float_safe(match.group(1))
            time_type = match.group(2).lower()
            
            if time_type == 'cpu':
                breakdown['cpu_time'] = time_value
            elif time_type == 'user':
                breakdown['user_time'] = time_value
            elif time_type == 'system':
                breakdown['system_time'] = time_value
        
        logger.info(f"Parsed prefill breakdown from old format: user={breakdown['user_time']}, system={breakdown['system_time']}, cpu={breakdown['cpu_time']}")
    
    return breakdown

## from Perf or Rusage
def __parse_first_decode_breakdown(content: str) -> Dict[str, float]:
    """
    Parse the first decode step time breakdown from both old and new log content formats.
    
    Returns a dictionary with the following keys:
    - 'cpu_time': CPU time for first decode
    - 'user_time': User time for first decode
    - 'system_time': System time for first decode
    """
    breakdown = {
        'cpu_time': 0.0,
        'user_time': 0.0,
        'system_time': 0.0
    }
    
    # Try new format first
    new_format_pattern = r"Step 0:\s*\n  Wall clock time: \d+ ms\s*\n  User time: ([\d.e-]+) sec\s*\n  System time: ([\d.e-]+) sec\s*\n  Total CPU time \(user\+system\): ([\d.e-]+) sec"
    new_match = re.search(new_format_pattern, content)
    
    if new_match:
        # Extract values from the new format
        breakdown['user_time'] = __convert_str2float_safe(new_match.group(1))
        breakdown['system_time'] = __convert_str2float_safe(new_match.group(2))
        breakdown['cpu_time'] = __convert_str2float_safe(new_match.group(3))
        logger.info(f"Parsed first decode breakdown from new format: user={breakdown['user_time']}, system={breakdown['system_time']}, cpu={breakdown['cpu_time']}")
        return breakdown
    
    # If new format not found, try old format
    # Find the line with "Decode 0 took" to locate the first decode section
    decode_marker = "Decode 0 took"
    decode_pos = content.find(decode_marker)
    
    if decode_pos != -1:
        # Extract a reasonable chunk of text after the marker
        section_end = content.find("Decode 1", decode_pos)
        if section_end == -1:  # If no "Decode 1" marker, take a reasonable amount
            section_end = decode_pos + 200
        
        decode_section = content[decode_pos:section_end]
        
        # Find all time measurements in this section using the pattern "- X [sec] Type time"
        time_pattern = r"-\s*([\d.]+)\s*\[sec\] (CPU|User|System) time"
        for match in re.finditer(time_pattern, decode_section):
            time_value = __convert_str2float_safe(match.group(1))
            time_type = match.group(2).lower()
            
            if time_type == 'cpu':
                breakdown['cpu_time'] = time_value
            elif time_type == 'user':
                breakdown['user_time'] = time_value
            elif time_type == 'system':
                breakdown['system_time'] = time_value
        
        logger.info(f"Parsed first decode breakdown from old format: user={breakdown['user_time']}, system={breakdown['system_time']}, cpu={breakdown['cpu_time']}")
    
    return breakdown

def analyze_log_file(file_path: str) -> Optional[LogMetrics]:
    metrics = LogMetrics()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parse all components
        metrics.phases = __parse_phase_info(content)
        metrics.decode_times = __parse_decode_time(content)
        metrics.prefill_time, metrics.time_to_first_token = __parse_prefill_and_ttft(content)
        
        # Parse decoding time breakdown
        metrics.decoding_time_breakdown = __parse_decode_time_breakdown(content)
        
        # Parse prefill and first decode breakdowns
        metrics.prefill_time_breakdown = __parse_prefill_breakdown(content)
        metrics.first_decode_breakdown = __parse_first_decode_breakdown(content)
        
        # Modified validation logic
        # For new format, we don't strictly require 'Decode' in phases
        # as long as we have valid decode times and TTFT
        if 'Decode' not in metrics.phases:
            logger.warning(f"No Decode phase information found in {file_path}")
            # Check if we have decode times and TTFT, which would indicate we're using the new format
            if metrics.decode_times and metrics.time_to_first_token > 0:
                logger.info(f"Using new format validation for {file_path} - decode times and TTFT are valid")
                # Create a dummy Decode phase entry if needed
                decode_metrics = PhaseMetrics()
                decode_metrics.duration = metrics.time_to_first_token - metrics.prefill_time
                metrics.phases['Decode'] = decode_metrics
            else:
                metrics.is_valid = False
        
        return metrics
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        return None

def process_directory(base_dir: str, ram_dirs: Dict[str, str], input_tokens: List[int]) -> Dict:
    results = defaultdict(lambda: defaultdict(list))
    valid_data_found = False
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist")
        return results
    
    for ram_size, dir_name in ram_dirs.items():
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found for RAM size {ram_size}: {dir_path}")
            continue
            
        for token in input_tokens:
            pattern = f"output_{token}_\\d+\\.(txt|log)"
            matching_files = [f for f in os.listdir(dir_path) if re.match(pattern, f)]
            
            if not matching_files:
                logger.warning(f"No matching files found for token {token} in {ram_size}")
                continue
                
            valid_files = 0
            for file_name in matching_files:
                logger.info(f"=====================================")
                logger.info(f"Processing file {file_name}")
                metrics = analyze_log_file(os.path.join(dir_path, file_name))

                if metrics and metrics.is_valid:
                    results[ram_size][token].append(metrics)
                    valid_files += 1
                    valid_data_found = True
            
            if valid_files > 0:
                logger.info(f"Processed {valid_files} valid files for {ram_size}, token {token}")
            logger.info(f"=====================================\n")
    
    if not valid_data_found:
        logger.error("No valid data found in any of the directories")
    
    return results

############################################
# Plotting Functions
############################################


def create_ttft_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect prefill and first decode data for this token length
        prefill_values = []
        first_decode_values = []
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid]
                if metrics_list:
                    # Calculate average prefill time (convert from ms to sec)
                    avg_prefill = np.mean([m.prefill_time for m in metrics_list]) / 1000
                    
                    # Get first decode time from decode_times list
                    avg_first_decode = np.mean([m.time_to_first_token if m.time_to_first_token else 0 for m in metrics_list]) / 1000
                    
                    prefill_values.append(avg_prefill)
                    first_decode_values.append(avg_first_decode)
                    token_has_data = True
                    has_data = True
                else:
                    prefill_values.append(None)
                    first_decode_values.append(None)
            else:
                prefill_values.append(None)
                first_decode_values.append(None)
        
        if token_has_data:
            # Add prefill bars
            fig.add_trace(go.Bar(
                name='Prefill',
                x=ram_sizes,
                y=prefill_values,
                marker_color='#ff7f0e',
                text=[f"{v:.2f}" if v is not None else "N/A" for v in prefill_values],
                textposition='auto'
            ))
            
            # Add first decode bars
            fig.add_trace(go.Bar(
                name='Decode',
                x=ram_sizes,
                y=first_decode_values,
                marker_color='#1f77b4',
                text=[f"{v:.2f}" if v is not None else "N/A" for v in first_decode_values],
                textposition='auto'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Time to First Token Breakdown (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Time (seconds)",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                # legend_title="Components",
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=0.1,
                    zerolinecolor='LightGray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for TTFT plot")
        return None

    return figs

def create_ttft_detailed_breakdown_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # Define colors for each component
    colors = {
        'prefill_user': '#FFE692',    # Light Salmon
        'prefill_system': '#D8D8D8',  # Coral
        'prefill_other': '#FFB0A5',   # Deep Orange
        'decode_user': '#FFE692',     # Light Blue
        'decode_system': '#D8D8D8',   # Light Blue
        'decode_other': '#FFB0A5'     # Blue
    }
    
    border_color = 'black'
    border_width = 0.5
    font_size = 16
    
    prefill_patterns = {
        'prefill_user': '/',
        'prefill_system': '/',
        'prefill_other': '/'
    }

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Initialize data containers
        data = {
            'prefill_user': [],
            'prefill_system': [],
            'prefill_other': [],
            'decode_user': [],
            'decode_system': [],
            'decode_other': []
        }
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid]
                if metrics_list:
                    # Calculate average prefill time (convert from ms to sec)
                    avg_prefill = np.mean([m.prefill_time for m in metrics_list]) / 1000
                    
                    # Get first decode time from decode_times list
                    avg_decode = np.mean([m.time_to_first_token if m.time_to_first_token else 0 for m in metrics_list]) / 1000
                    
                    # Calculate prefill components
                    avg_prefill_user = np.mean([m.prefill_time_breakdown['user_time'] for m in metrics_list])
                    avg_prefill_system = np.mean([m.prefill_time_breakdown['system_time'] for m in metrics_list])
                    avg_prefill_cpu = np.mean([m.prefill_time_breakdown['cpu_time'] for m in metrics_list])
                    
                    # Calculate "other" time for prefill (CPU time that's not user or system time)
                    avg_prefill_other = max(0, avg_prefill - avg_prefill_user - avg_prefill_system)
                    
                    # Calculate first decode components
                    avg_decode_user = np.mean([m.first_decode_breakdown['user_time'] for m in metrics_list])
                    avg_decode_system = np.mean([m.first_decode_breakdown['system_time'] for m in metrics_list])
                    avg_decode_cpu = np.mean([m.first_decode_breakdown['cpu_time'] for m in metrics_list])
                    
                    # Calculate "other" time for first decode
                    avg_decode_other = max(0, avg_decode - avg_decode_user - avg_decode_system)
                    
                    # Store values
                    data['prefill_user'].append(avg_prefill_user)
                    data['prefill_system'].append(avg_prefill_system)
                    data['prefill_other'].append(avg_prefill_other)
                    data['decode_user'].append(avg_decode_user)
                    data['decode_system'].append(avg_decode_system)
                    data['decode_other'].append(avg_decode_other)
                    
                    token_has_data = True
                    has_data = True
                else:
                    for key in data:
                        data[key].append(None)
            else:
                for key in data:
                    data[key].append(None)
        
        if token_has_data:
            # Add traces for each component in stacked order
            component_order = [
                ('prefill_user', 'CPU Time in Prefill'),
                ('prefill_system', 'I/O Time in Prefill'),
                ('prefill_other', 'Preemption Delay in Prefill'),
                ('decode_user', 'CPU Time in Decode'),
                ('decode_system', 'I/O Time in Decode'),
                ('decode_other', 'Preemption Delay in Decode')
            ]
            
            for key, name in component_order:
                if key.startswith('prefill'):
                    fig.add_trace(go.Bar(
                        name=name,
                        x=ram_sizes,
                        y=data[key],
                        marker=dict(
                            color=colors[key],
                            pattern=dict(
                                shape=prefill_patterns[key],
                                solidity=0.3  # Adjust pattern density
                            ),
                            line=dict(color=border_color, width=border_width),
                        ),
                        text=[f"{v:.2f}" if v is not None and v > 0.01 else "" for v in data[key]],
                        textposition='inside',
                        textfont=dict(
                            size=font_size,           # Font size for bar values
                            color='black',     # Font color for bar values
                            family='Arial'     # Font family
                        ),
                    ))
                else:
                    fig.add_trace(go.Bar(
                    name=name,
                    x=ram_sizes,
                    y=data[key],
                    marker=dict(
                        color=colors[key],
                        line=dict(color=border_color, width=border_width),
                    ),
                    # marker_line=dict(color=border_color, width=border_width),
                    text=[f"{v:.2f}" if v is not None and v > 0.01 else "" for v in data[key]],
                    textposition='inside',
                    textfont=dict(
                        size=font_size,           # Font size for bar values
                        color="#000000",     # Font color for bar values
                        family='Arial'     # Font family
                    ),
                ))

            # Update layout
            fig.update_layout(
                # title=f"Time to First Token Detailed Breakdown (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Time [seconds]",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                height=600,
                width=1000,
                # Legend customization
                legend=dict(
                    font=dict(
                        family="Arial",    # Font family
                        size=font_size-1,           # Font size for legend text
                        color="#000000"    # Font color for legend text
                    ),
                    # Additional legend customization options
                    bgcolor="rgba(255,255,255,0.8)",    # Legend background color with transparency
                    # bordercolor="Black",                # Legend border color
                    # borderwidth=1,                      # Legend border width
                    orientation="v",                    # Horizontal legend (use "v" for vertical)
                    xanchor="left",                    # Legend horizontal anchor point
                    yanchor="top",                   # Legend vertical anchor point
                    # y=1,                             # Legend y position (above the plot)
                    # x=1                                 # Legend x position
                ),
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGray'
                )
            )
            
            # Add a horizontal line separating prefill and decode components
            for i, ram_size in enumerate(ram_sizes):
                if all(data[f'prefill_{comp}'][i] is not None for comp in ['user', 'system', 'other']):
                    prefill_total = data['prefill_user'][i] + data['prefill_system'][i] + data['prefill_other'][i]
                    
                    fig.add_shape(
                        type="line",
                        x0=i-0.4, x1=i+0.4,
                        y0=prefill_total, y1=prefill_total,
                        line=dict(color="black", width=1.5, dash="dot")
                    )
            
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for TTFT detailed breakdown plot")
        return None

    return figs

def create_decode_latency_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect data for this token length
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and m.decode_times]
                if metrics_list:
                    # Calculate average decode time
                    avg_decode = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list])
                    y_values.append(avg_decode)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:.2f}" if v is not None else "N/A" for v in y_values],
                textposition='outside',
                marker_color='#1f77b4'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Average Decoding Latency (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Decode Time (seconds)",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='lightgray',
                    zerolinewidth=1,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=0.1,
                    zerolinecolor='LightGray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decode latency plot")
        return None

    return figs

def create_decoding_time_breakdown_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    """
    Create a stacked bar plot showing breakdown of decoding time 
    across different components for different RAM sizes.
    """
    figs = []
    has_data = False

    # Color scheme for time components
    time_colors = {
        'user_time': '#FFE692',      
        'system_time': '#D8D8D8',    
        'other_time': '#FFB0A5'      
    }
    
    border_color = 'black'
    border_width = 0.5

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Prepare data for time components
        time_values = {
            'avg_latency': [],
            'user_time': [],
            'system_time': [],
            'other_time': []
        }
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid]
                if metrics_list:
                    # Calculate averages for latency, user time, and system time
                    avg_latency = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list])
                    logger.info(f"AVG {avg_latency}")
                    avg_user_time = np.mean([m.decoding_time_breakdown['user_time'] for m in metrics_list]) / 1
                    avg_system_time = np.mean([m.decoding_time_breakdown['system_time'] for m in metrics_list]) / 1
                    
                    # Calculate other time
                    avg_cpu_time = avg_user_time + avg_system_time
                    avg_other_time = max(0, avg_latency - avg_cpu_time)
                    # avg_latency = 
                    
                    # Store values
                    time_values['avg_latency'].append(avg_latency)
                    time_values['user_time'].append(avg_user_time)
                    time_values['system_time'].append(avg_system_time)
                    time_values['other_time'].append(avg_other_time)
                    
                    token_has_data = True
                    has_data = True
                else:
                    for key in time_values:
                        time_values[key].append(None)
            else:
                for key in time_values:
                    time_values[key].append(None)
        
        if token_has_data:
            
            component_order = [
                ('prefill_user', 'CPU Time in Prefill'),
                ('prefill_system', 'I/O Time in Prefill'),
                ('prefill_other', 'Preemption Delay in Prefill'),
                ('decode_user', 'CPU Time in Decode'),
                ('decode_system', 'I/O Time in Decode'),
                ('decode_other', 'Preemption Delay in Decode')
            ]
            
            # Add traces for user time and system time
            fig.add_trace(go.Bar(
                name='CPU Time',
                x=ram_sizes,
                y=time_values['user_time'],
                marker=dict(
                    color=time_colors['user_time'],
                    line=dict(color=border_color, width=border_width),
                ),
                text=[f"{v:.2f}" if v is not None else "N/A" for v in time_values['user_time']],
                textposition='inside',
                textfont=dict(
                    size=15,           # Font size for bar values
                    color='black',     # Font color for bar values
                    family='Arial'     # Font family
                ),
            ))
            
            fig.add_trace(go.Bar(
                name='I/O Time',
                x=ram_sizes,
                y=time_values['system_time'],
                marker=dict(
                    color=time_colors['system_time'],
                    line=dict(color=border_color, width=border_width),
                ),
                text=[f"{v:.2f}" if v is not None else "N/A" for v in time_values['system_time']],
                textposition='inside',
                textfont=dict(
                    size=15,           # Font size for bar values
                    color='black',     # Font color for bar values
                    family='Arial'     # Font family
                ),
            ))
            
            # Add other time trace
            fig.add_trace(go.Bar(
                name='Preemption Delay',
                x=ram_sizes,
                y=time_values['other_time'],
                marker=dict(
                    color=time_colors['other_time'],
                    line=dict(color=border_color, width=border_width),
                ),
                text=[f"{v:.2f}" if v is not None else "N/A" for v in time_values['other_time']],
                textposition='inside',
                textfont=dict(
                    size=15,           # Font size for bar values
                    color='black',     # Font color for bar values
                    family='Arial'     # Font family
                ),
            ))

            # Update layout
            fig.update_layout(
                title=f"Decoding Time Breakdown (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Time (seconds)",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                height=600,
                width=1000,
                # Legend customization
                legend=dict(
                    font=dict(
                        family="Arial",    # Font family
                        size=15,           # Font size for legend text
                        color="#000000"    # Font color for legend text
                    ),
                    # Additional legend customization options
                    bgcolor="rgba(255,255,255,0.8)",    # Legend background color with transparency
                    # bordercolor="Black",                # Legend border color
                    # borderwidth=1,                      # Legend border width
                    orientation="v",                    # Horizontal legend (use "v" for vertical)
                    xanchor="left",                    # Legend horizontal anchor point
                    yanchor="top",                   # Legend vertical anchor point
                    # y=1,                             # Legend y position (above the plot)
                    # x=1                                 # Legend x position
                ),
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='lightgray',
                )
            )
            
            # Add total time annotation
            fig.add_trace(go.Scatter(
                x=ram_sizes,
                y=time_values['avg_latency'],
                mode='text',
                # text=[f"Total: {v:.2f}" for v in time_values['avg_latency']],
                textposition='top center',
                showlegend=False
            ))
            
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decoding time breakdown plot")
        return None

    return figs

############################################

def main():
    
    parser = argparse.ArgumentParser(description='Analyze inference logs for different RAM sizes.')
    parser.add_argument('--base-dir', type=str, required=True,
                      help='Base directory containing RAM size subdirectories')
    parser.add_argument('--ram-sizes', type=str, nargs='+',
                      help='List of RAM sizes to analyze (e.g., 4G 8G 16G 512M)')
    parser.add_argument('--input-tokens', type=int, nargs='+', default=[8],
                      help='List of input token lengths to analyze')
    parser.add_argument('--export-csv', type=bool, default=True,
                      help='Export results to a CSV file')
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.isdir(args.base_dir):
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return

    # Get available RAM sizes if not specified
    if args.ram_sizes is None:
        available_ram_sizes = get_available_ram_sizes(args.base_dir)
        if not available_ram_sizes:
            logger.error("No valid RAM size directories found")
            return
        logger.info(f"Found RAM sizes: {', '.join(available_ram_sizes)}")
        ram_sizes = available_ram_sizes
    else:
        ram_sizes = args.ram_sizes
        
    # Convert RAM sizes to directory names
    ram_dirs = {size: get_ram_dir_name(size) for size in ram_sizes}
    
    logger.info("Starting log analysis...")
    logger.info(f"Analyzing RAM sizes: {', '.join(ram_sizes)}")
    logger.info(f"Input tokens: {args.input_tokens}")
    
    # Process all log files
    results = process_directory(args.base_dir, ram_dirs, args.input_tokens)
    
    # export results to a csv file
    if args.export_csv:
        export_results_to_csv(results)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create and display plots
    plot_functions = {
        # 'ttft_breakdown': create_ttft_plot,
        # 'ttft_detailed_breakdown': create_ttft_detailed_breakdown_plot,
        # 'decode_latency': create_decode_latency_plot,
        'decoding_time_breakdown': create_decoding_time_breakdown_plot,
    }
    
    plots_created = False
    for name, func in plot_functions.items():
        figs = func(results, ram_sizes, args.input_tokens)
        if figs is not None:
            for token, fig in figs:
                # Save as PNG
                png_path = os.path.join('plots', f'{name}_{token}_tokens.png')
                fig.write_image(png_path)
                logger.info(f"Created plot: {png_path}")
                
                # Show the plot in browser
                fig.show()
            plots_created = True
    
    if plots_created:
        logger.info("Analysis complete. Plots have been saved to the 'plots' directory.")
    else:
        logger.error("No plots were created due to lack of valid data.")

if __name__ == "__main__":
    main()