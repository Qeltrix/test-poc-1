#!/usr/bin/env python3
"""
Qeltrix (.qltx) - Content-derived, parallel, streaming obfuscation container (PoC)

Copyright (c) 2025 @hejhdiss(Muhammed Shafin P)
All rights reserved.
Licensed under GPLv3.
"""
import os
import sys
import subprocess
import tempfile
import pathlib
import filecmp
import time
import json
import secrets

# Libraries required for advanced features (must be installed via 'pip install lz4 matplotlib numpy')
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] Matplotlib and/or NumPy not installed. Install with: pip install matplotlib numpy")
    sys.exit(1)

# --- Configuration ---
# Locate qeltrix.py in the same directory as this test script using __file__
QELTRIX_SCRIPT = pathlib.Path(__file__).resolve().parent / "qeltrix.py"
# The Python executable to run the script
PYTHON_EXECUTABLE = sys.executable
VERSION_TAG = "QLTX-V1"

# --- Data Structures ---
TestResult = dict[str, any]

def calculate_entropy(filepath: pathlib.Path) -> float:
    """Calculates the Shannon Entropy (bits/byte) of a file's content using NumPy."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return 0.0
    
    try:
        data = filepath.read_bytes()
        # Count byte frequencies
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        
        # Calculate probabilities
        probabilities = counts / len(data)
        
        # Filter out zero probabilities
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    except Exception as e:
        print(f"Warning: Could not calculate entropy for {filepath.name}: {e}")
        return 0.0

def create_dummy_file(filepath: pathlib.Path, file_type: str, size_mb: float) -> None:
    """Creates a dummy file of a specified type and size."""
    # Convert MB to bytes, but handle specific, non-clean byte sizes for edge cases
    if file_type == "Non-Multiple Block Size Stress":
        # Create a file size that is NOT a clean multiple of 1MB, e.g., 8,400,000 bytes (~8.01 MB)
        size_bytes = 8400000 
        file_type = "Low Compressibility Binary" # Base type for this test
    else:
        size_bytes = int(size_mb * 1024 * 1024)
        
    print(f"-> Creating input file '{filepath.name}' (Type: {file_type}, Target Size: {size_bytes / (1024*1024):.2f} MB)")
    
    # Content creation logic 
    try:
        if file_type == "Highly Compressible Text":
            content = "QELTRIX TEST DATA " * 1000
            repeat_count = size_bytes // len(content.encode("utf-8")) + 1
            data = (content * repeat_count)[:size_bytes]
            filepath.write_text(data, encoding="utf-8")
        
        elif file_type == "Low Compressibility Binary":
            random_data = secrets.token_bytes(size_bytes)
            filepath.write_bytes(random_data)
            
        elif file_type == "Zeroed Data":
            filepath.write_bytes(b'\x00' * size_bytes)
            
        elif file_type == "Empty":
            filepath.write_bytes(b'')

        elif file_type == "JSON Data":
            data = {f"key_{i}": f"value_content_long_string_{i}" * 10 for i in range(100)}
            content = json.dumps(data, indent=4)
            repeat_count = size_bytes // len(content.encode("utf-8")) + 1
            data_to_write = (content * repeat_count)[:size_bytes]
            filepath.write_text(data_to_write, encoding="utf-8")
        
        actual_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   Actual Size: {actual_size_mb:.2f} MB")
        
    except IOError as e:
        print(f"Error creating file {filepath.name}: {e}")
        sys.exit(1)

def run_command(command: list[str], description: str) -> tuple[subprocess.CompletedProcess, float]:
    """Executes a command, returns the completed process result and time taken."""
    full_cmd = [PYTHON_EXECUTABLE, str(QELTRIX_SCRIPT)] + command
    
    print(f"\n--- Running Command: {description} ---")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            full_cmd, 
            check=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            encoding="utf-8",
            timeout=300 # Increased timeout for safety with larger files
        )
        time_taken = time.time() - start_time
        print(f"Status: SUCCESS (Exit Code {result.returncode})")
        print(f"Time taken: {time_taken:.2f} seconds")
        return result, time_taken

    except subprocess.CalledProcessError as e:
        print(f"\nStatus: FAILED (Exit Code {e.returncode})")
        raise Exception(f"Command failed: {description}") from e
    except Exception as e:
        raise Exception(f"Command execution error: {e}")

def verify_files(file1: pathlib.Path, file2: pathlib.Path) -> None:
    """Compares two files and raises an error if they differ."""
    if not file1.exists() or not file2.exists():
        raise FileNotFoundError(f"Verification FAILED: One or both files are missing: {file1.name}, {file2.name}")

    if not filecmp.cmp(file1, file2, shallow=False):
        raise ValueError(f"Verification FAILED: Files {file1.name} and {file2.name} differ (content mismatch).")
    
    print(f"Verification SUCCESS: {file1.name} and {file2.name} are identical.")

def run_single_test(test_config: dict, temp_dir: pathlib.Path) -> TestResult:
    """Runs a single pack/unpack test based on configuration and returns metrics."""
    test_config.setdefault('no_permute', False)
    
    test_id = f"{VERSION_TAG}-{test_config['file_type']}-{test_config['mode']}"
    if test_config['no_permute']:
        test_id += "-NOP"
    if 'block_size_kb' in test_config:
        test_id += f"-B{test_config['block_size_kb']}"
    if 'head_bytes' in test_config and test_config['head_bytes'] < (1<<20):
        test_id += f"-H{test_config['head_bytes'] // 1024}K"


    input_file_name = f"input_{test_id}.dat"
    packed_file_name = f"packed_{test_id}.qltx"
    unpacked_file_name = f"unpacked_{test_id}.dat"

    input_file = temp_dir / input_file_name
    packed_file = temp_dir / packed_file_name
    unpacked_file = temp_dir / unpacked_file_name
    
    # Block size is stored in bytes for the command line
    block_size_bytes = test_config.get('block_size', 1<<20)

    results = {
        'Test ID': test_id,
        'File Type': test_config['file_type'],
        'Mode': test_config['mode'],
        'Block Size (KB)': block_size_bytes // 1024,
        'No Permute': test_config['no_permute'],
        'Pack Time (s)': 0.0,
        'Unpack Time (s)': 0.0,
        'Pack Throughput (MB/s)': 0.0,
        'Unpack Throughput (MB/s)': 0.0,
        'Original Size (MB)': 0.0,
        'Packed Size (MB)': 0.0,
        'Input Entropy (bits/byte)': 0.0,
        'Output Entropy (bits/byte)': 0.0,
        'Ratio (%)': 0.0,
        'Status': 'PENDING'
    }

    try:
        # 1. Create Test Data
        create_dummy_file(input_file, test_config['file_type'], test_config['size_mb'])
        
        original_size_mb = input_file.stat().st_size / (1024 * 1024)
        results['Original Size (MB)'] = original_size_mb
        results['Input Entropy (bits/byte)'] = calculate_entropy(input_file)

        # 2. Pack Command
        pack_cmd = [
            "pack", 
            str(input_file), 
            str(packed_file),
            "--mode", test_config['mode'],
            "--block-size", str(block_size_bytes),
            *(["--head-bytes", str(test_config['head_bytes'])] if test_config['mode'] == 'single_pass_firstN' else []),
            *(["--no-permute"] if test_config['no_permute'] else [])
        ]
        _, pack_time = run_command(pack_cmd, f"Pack: {test_id}")
        results['Pack Time (s)'] = pack_time
        results['Pack Throughput (MB/s)'] = original_size_mb / pack_time if pack_time > 0 else 0
        
        if packed_file.exists():
            results['Packed Size (MB)'] = packed_file.stat().st_size / (1024 * 1024)
            results['Output Entropy (bits/byte)'] = calculate_entropy(packed_file)
            results['Ratio (%)'] = (results['Packed Size (MB)'] / results['Original Size (MB)']) * 100 if results['Original Size (MB)'] > 0 else 0
        else:
            raise FileNotFoundError("Packed file was not created.")

        # 3. Unpack Command
        unpack_cmd = ["unpack", str(packed_file), str(unpacked_file)]
        _, unpack_time = run_command(unpack_cmd, f"Unpack: {test_id}")
        results['Unpack Time (s)'] = unpack_time
        results['Unpack Throughput (MB/s)'] = original_size_mb / unpack_time if unpack_time > 0 else 0

        # 4. Verify Result
        verify_files(input_file, unpacked_file)
        results['Status'] = 'PASS'
        
    except Exception as e:
        print(f"\n!!! TEST FAILED: {test_id} - {e} !!!")
        results['Status'] = f'FAIL: {type(e).__name__}'

    return results

def display_results_table(all_results: list[TestResult]) -> tuple[str, list[dict]]:
    """Formats and prints the test results as a string table and returns it."""
    
    # Define the columns and their formatting (key, format_spec, min_width)
    columns = [
        ('Test ID', '<', 30), 
        ('Original Size (MB)', '.2f', 10), 
        ('Packed Size (MB)', '.2f', 10), 
        ('Ratio (%)', '.1f', 8), 
        ('Input Entropy (bits/byte)', '.4f', 10),
        ('Output Entropy (bits/byte)', '.4f', 10),
        ('Mode', '<', 18),
        ('No Permute', '<', 10),
        ('Pack Time (s)', '.2f', 8), 
        ('Unpack Time (s)', '.2f', 8), 
        ('Pack Throughput (MB/s)', '.1f', 12),
        ('Unpack Throughput (MB/s)', '.1f', 12),
        ('Status', '<', 15)
    ]
    
    # Calculate column widths (minimum width enforced by third tuple item)
    col_widths = [c[2] for c in columns]
    header = [col[0] for col in columns]

    # Recalculate widths based on data content
    for result in all_results:
        for i, (key, fmt, min_width) in enumerate(columns):
            value = result.get(key, 'N/A')
            
            display_value = str(value)
            if key == 'No Permute':
                display_value = 'Yes' if value else 'No'
            elif fmt != '<':
                 # Apply format to number
                try:
                    display_value = f"{value:{fmt}}"
                except:
                    display_value = str(value)
            
            # Update the width based on the rendered string length
            col_widths[i] = max(col_widths[i], len(display_value))

    # Generate output string
    separator = " | "
    total_width = sum(col_widths) + len(separator) * (len(columns) - 1)
    
    output = []
    output.append("=" * total_width)
    output.append(" " * (total_width // 2 - 20) + "QELTRIX V1 INTEGRATION TEST SUMMARY RESULTS")
    output.append("=" * total_width)

    # Header
    header_line = separator.join(h.center(w) for h, w in zip(header, col_widths))
    output.append(header_line)
    output.append("-|-".join("-" * w for w in col_widths))

    # Data Rows
    for result in all_results:
        row = []
        for key, fmt, min_width in columns:
            value = result.get(key, 'N/A')
            
            if key == 'No Permute':
                display_value = 'Yes' if value else 'No'
                display_value = display_value.ljust(col_widths[columns.index((key, fmt, min_width))])
            elif fmt == '<':
                display_value = str(value).ljust(col_widths[columns.index((key, fmt, min_width))])
            else:
                try:
                    # Fix: Use the correct column definition tuple for lookup
                    idx = columns.index((key, fmt, min_width))
                    display_value = f"{value:{fmt}}".rjust(col_widths[idx])
                except:
                    # Fallback in case of unexpected format or lookup failure
                    idx = columns.index((key, fmt, min_width))
                    display_value = str(value).rjust(col_widths[idx])
                    
            row.append(display_value)
        output.append(separator.join(row))
    
    output.append("=" * total_width)
    output.append("NOTE: Ratio (%) is Packed Size / Original Size (Lower is better compression).")
    output.append(f"Maximum possible output entropy is 8.0 bits/byte (Closer to 8.0 is better obfuscation).")
    output.append(f"Throughput (MB/s) is Original Size / Time (Higher is better performance).")
    output.append(f"Copyright @hejhdiss(Muhammed Shafin P)")
    output.append("=" * total_width)
    
    txt_table = "\n".join(output)
    
    print("\n" + txt_table)
    
    # Return a list of dicts that are safe for JSON serialization
    json_safe_results = []
    for r in all_results:
        # Create a deep copy and sanitize
        safe_r = r.copy()
        safe_r['No Permute'] = 'Yes' if safe_r['No Permute'] else 'No'
        json_safe_results.append(safe_r)
        
    return txt_table, json_safe_results # Return raw dict list for JSON

def save_results_to_files(txt_table: str, json_data: list[dict], base_path: pathlib.Path) -> None:
    """Saves the results table to TXT and JSON files."""
    
    # 1. Save TXT
    txt_filepath = base_path / "results_summary.txt"
    try:
        txt_filepath.write_text(txt_table, encoding="utf-8")
        print(f"\n[FILE SAVED] Plain text table summary: {txt_filepath.name}")
    except Exception as e:
        print(f"[ERROR] Could not save TXT file: {e}")

    # 2. Save JSON
    json_filepath = base_path / "results_summary.json"
    try:
        json_filepath.write_text(json.dumps(json_data, indent=4), encoding="utf-8")
        print(f"[FILE SAVED] JSON data summary: {json_filepath.name}")
    except Exception as e:
        print(f"[ERROR] Could not save JSON file: {e}")


def plot_results(all_results: list[TestResult], base_path: pathlib.Path) -> None:
    """Generates a Matplotlib plot comparing compression ratio, pack time, unpack time, and entropy."""
    
    plot_data = [r for r in all_results if r['Original Size (MB)'] > 0 and r['Status'] == 'PASS']
    if not plot_data:
        print("\nCould not generate plot: No successful, non-empty tests available.")
        return

    # Helper function for cleaner labels
    def clean_label(label):
        return label.replace(f"{VERSION_TAG}-", "").replace(" Compressibility ", "\n").replace("-", "\n").replace("Block Size Stress", "Block\nStress")
        
    test_labels = [clean_label(r['Test ID']) for r in plot_data]
    ratios = [r['Ratio (%)'] for r in plot_data]
    pack_times = [r['Pack Time (s)'] for r in plot_data]
    unpack_times = [r['Unpack Time (s)'] for r in plot_data]
    pack_throughput = [r['Pack Throughput (MB/s)'] for r in plot_data]
    unpack_throughput = [r['Unpack Throughput (MB/s)'] for r in plot_data]
    out_entropy = [r['Output Entropy (bits/byte)'] for r in plot_data]
    
    x = np.arange(len(test_labels))  # the label locations
    width = 0.2  # the width of the bars

    # Create a figure and the first axis
    fig, ax1 = plt.subplots(figsize=(20, 10)) # Wider figure for more axes

    # --- AXIS 1 (Left Primary): Compression Ratio (Bar Chart) ---
    # Give rects1 an explicit label
    rects1 = ax1.bar(x - width*1.5, ratios, width, label='Compression Ratio (%) (Lower is Better)', color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Compression Ratio (Packed/Original %)', color='#2ecc71', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.set_ylim(0, max(max(ratios) * 1.1, 105) if ratios else 105) 
    # Give axhline an explicit label
    line_ratio_ref, = ax1.plot(ax1.get_xlim(), [100, 100], color='r', linestyle='--', linewidth=0.8, label='100% (No Compression)')


    # --- AXIS 2 (Left Secondary): Output Entropy (Bar Chart) ---
    ax3 = ax1.twinx() 
    ax3.spines['right'].set_position(('outward', 0)) # Position it on the left side
    # Give rects3 an explicit label
    rects3 = ax3.bar(x - width*0.5, out_entropy, width, label='Output Entropy (bits/byte) (Closer to 8.0 is Better)', color='#f39c12', alpha=0.8)
    ax3.set_ylabel('Output Entropy (bits/byte)', color='#f39c12', fontsize=12, fontweight='bold', rotation=270, labelpad=15)
    ax3.tick_params(axis='y', labelcolor='#f39c12')
    ax3.set_ylim(0, 8.5) # Max entropy is 8.0
    # Give axhline an explicit label
    line_entropy_ref, = ax3.plot(ax3.get_xlim(), [8.0, 8.0], color='k', linestyle=':', linewidth=0.8, label='Max Entropy (8.0)')
    
    # --- AXIS 3 (Right Primary): Pack and Unpack Time (Line Chart) ---
    ax2 = ax1.twinx() 
    # Pack Time line
    line1, = ax2.plot(x + width/2, pack_times, color='#3498db', marker='o', linestyle='-', linewidth=2, label='Pack Time (s) (Lower is Better)')
    # Unpack Time line 
    line2, = ax2.plot(x + width/2, unpack_times, color='#9b59b6', marker='s', linestyle='--', linewidth=2, label='Unpack Time (s) (Lower is Better)')
    
    ax2.set_ylabel('Time (Seconds)', color='#34495e', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#34495e')
    # Use the maximum of both pack and unpack times for the upper limit
    max_time = max(pack_times + unpack_times) if pack_times or unpack_times else 1.0
    ax2.set_ylim(0, max_time * 1.5)

    # --- AXIS 4 (Right Secondary): Pack and Unpack Throughput (Line Chart) ---
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 60)) # Offset the fourth axis
    
    # Pack Throughput line
    line3, = ax4.plot(x + width, pack_throughput, color='#e74c3c', marker='^', linestyle='-', linewidth=2, label='Pack Throughput (MB/s) (Higher is Better)')
    # Unpack Throughput line
    line4, = ax4.plot(x + width, unpack_throughput, color='#1abc9c', marker='v', linestyle='--', linewidth=2, label='Unpack Throughput (MB/s) (Higher is Better)')
    
    ax4.set_ylabel('Throughput (MB/s)', color='#c0392b', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#c0392b')
    # Set Y limit for throughput
    max_throughput = max(pack_throughput + unpack_throughput) if pack_throughput or unpack_throughput else 1.0
    ax4.set_ylim(0, max_throughput * 1.5)
    
    # --- General Plot Styling and Labels ---
    title_text = f'{VERSION_TAG} Performance, Compression, and Obfuscation Analysis'
    subtitle_text = "Metrics Guide: For better results, look for LOW Compression Ratio (Green Bars), HIGH Output Entropy (Orange Bars), and HIGH Throughput (Red/Teal Lines)."
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.9, subtitle_text, ha="center", fontsize=10, color="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels(test_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_xlabel("Test Configuration (File Type | Mode | Block Size | Permutation)", fontsize=12)
    
    # --- Combined Legend (FIXED) ---
    # Collect all the handles and labels explicitly from the objects we created
    handles_list = [
        rects1[0], line_ratio_ref,
        rects3[0], line_entropy_ref,
        line1, line2,
        line3, line4,
    ]
    labels_list = [h.get_label() for h in handles_list]
    
    # Create the single legend
    ax1.legend(handles_list, labels_list, 
               loc='upper left', 
               bbox_to_anchor=(0.0, 1.0), 
               ncol=3, # Use 3 columns to save vertical space
               fontsize=9,
               framealpha=0.9)

    plt.grid(axis='y', alpha=0.3)
    # Adjusted rect to make space for copyright, subtitle, and multi-column legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.88]) 
    
    # Add Copyright Text
    plt.figtext(0.5, 0.01, "Copyright @hejhdiss(Muhammed Shafin P)", ha="center", fontsize=9, color="darkslategray")
    
    plot_filepath = base_path / "qeltrix_test_results.png"
    plt.savefig(plot_filepath)
    
    print(f"\n[FILE SAVED] Matplotlib Graph (PNG): {plot_filepath.name}")

def run_all_tests():
    """Main function to run all parameterized tests."""
    
    if not QELTRIX_SCRIPT.exists():
        print(f"Error: Required script not found at {QELTRIX_SCRIPT}")
        print("Please ensure 'qeltrix.py' is in the same folder as 'test_qeltrix.py'.")
        sys.exit(1)

    # --- Parameterized Test Configurations (Expanded & Hardened) ---
    # block_size is in bytes (1<<20 = 1MB)
    test_configs = [
        # 1. High-End Baseline (Compressible, Low Input Entropy) - 8MB
        {'file_type': 'Highly Compressible Text', 'size_mb': 8.0, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 2. High-End Incompressible Baseline (High Input Entropy) - 8MB
        {'file_type': 'Low Compressibility Binary', 'size_mb': 8.0, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 3. Structured Data / Fragmentation Test (2.5MB, Medium Block)
        {'file_type': 'JSON Data', 'size_mb': 2.5, 'mode': 'two_pass', 'block_size': 256 * 1024, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 4. Single Pass - Standard (Compressible, small block)
        {'file_type': 'Highly Compressible Text', 'size_mb': 3.0, 'mode': 'single_pass_firstN', 'block_size': 512 * 1024, 'head_bytes': 1<<20, 'no_permute': False}, 

        # 5. Permutation Impact Test (Compressible, NO PERMUTE)
        {'file_type': 'Highly Compressible Text', 'size_mb': 4.0, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': True},
        
        # 6. Block Size Overhead Test (Small file, VERY small blocks)
        {'file_type': 'Highly Compressible Text', 'size_mb': 0.5, 'mode': 'two_pass', 'block_size': 64 * 1024, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 7. Edge Case: Empty file (Tests zero-size handling)
        {'file_type': 'Empty', 'size_mb': 0.0, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 8. Single Pass, Low Compressibility, Large Head 
        {'file_type': 'Low Compressibility Binary', 'size_mb': 5.0, 'mode': 'single_pass_firstN', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': False},

        # 9. Low Compressibility Binary with Small Block Size (8MB, 256KB block) - Stress I/O
        {'file_type': 'Low Compressibility Binary', 'size_mb': 8.0, 'mode': 'two_pass', 'block_size': 256 * 1024, 'head_bytes': 1<<20, 'no_permute': False},
        
        # --- NEW HARDENED/STRESS TESTS ---

        # 10. Max Compression Case (Zeroed Data)
        {'file_type': 'Zeroed Data', 'size_mb': 4.0, 'mode': 'two_pass', 'block_size': 512 * 1024, 'head_bytes': 1<<20, 'no_permute': False},

        # 11. Large File Stress (16MB) - Pushes performance limits
        {'file_type': 'Low Compressibility Binary', 'size_mb': 16.0, 'mode': 'two_pass', 'block_size': 2<<20, 'head_bytes': 1<<20, 'no_permute': False}, # 2MB Block

        # 12. Very Small Block Size Stress (16KB) - Tests block index overhead
        {'file_type': 'Highly Compressible Text', 'size_mb': 8.0, 'mode': 'two_pass', 'block_size': 16 * 1024, 'head_bytes': 1<<20, 'no_permute': False}, # 16KB Block

        # 13. Non-Multiple Block Size Stress - Ensures correct padding/boundary handling (size is set in create_dummy_file)
        {'file_type': 'Non-Multiple Block Size Stress', 'size_mb': 8.0001, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': False},
        
        # 14. Minimal Head Bytes Stress - Stresses key derivation security with minimal material
        {'file_type': 'Low Compressibility Binary', 'size_mb': 2.0, 'mode': 'single_pass_firstN', 'block_size': 1<<20, 'head_bytes': 256, 'no_permute': False}, # 256 bytes head

        # 15. Two Pass, No Permute, Incompressible - Tests obfuscation security impact on random data
        {'file_type': 'Low Compressibility Binary', 'size_mb': 4.0, 'mode': 'two_pass', 'block_size': 1<<20, 'head_bytes': 1<<20, 'no_permute': True},
    ]

    all_results: list[TestResult] = []

    # Use a temporary directory for all test files
    base_path = QELTRIX_SCRIPT.parent
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        print(f"\nUsing temporary directory for test files: {temp_dir}")
        
        for config in test_configs:
            print("\n" + "#" * 80)
            print(f"STARTING TEST: {config['file_type']} - Mode: {config['mode']} - NoPermute: {config['no_permute']}")
            print("#" * 80)
            result = run_single_test(config, temp_dir)
            all_results.append(result)

    # 2. Display and Save Results
    txt_table, json_data = display_results_table(all_results)
    save_results_to_files(txt_table, json_data, base_path)
    plot_results(all_results, base_path) # Pass base_path to plot_results
    
    # 3. Final Overall Status Check
    failed_tests = [r for r in all_results if r['Status'].startswith('FAIL')]
    if failed_tests:
        print(f"\n!!! OVERALL TEST RUN FAILED: {len(failed_tests)} tests did not pass. !!!")
        sys.exit(1)
    else:
        print("\n$$$ ALL QELTRIX INTEGRATION TESTS PASSED SUCCESSFULLY $$$")

if __name__ == "__main__":
    # Check for test.py in the same directory and rename it to test_qeltrix.py if found.
    # This prevents the user from having to manually rename the file.
    test_file_path = pathlib.Path(__file__).resolve()
    if test_file_path.name == "test.py":
        new_path = test_file_path.parent / "test_qeltrix.py"
        try:
            test_file_path.rename(new_path)
            print(f"[NOTE] Renamed 'test.py' to 'test_qeltrix.py' for clarity.")
        except Exception as e:
            print(f"[WARNING] Could not rename test file: {e}")

    run_all_tests()