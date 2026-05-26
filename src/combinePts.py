# This script reads .pts files from a specified directory, processes them, and saves the results to a JSON file. It combines the pts files into a single JSON structure.

import os
import json
import glob


def read_pts_file(file_path):
    """Reads a .pts file and returns a list of (x, y) tuples."""
    points = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    x, y = map(float, line.strip().split())
                    points.append((x, y))
        return points
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except ValueError:
        print(f"Error: Invalid data format in file '{file_path}'. Expected two numbers per line.")
        return []

def generate_json_from_pts(directory, output_json):
    """Reads all kk*.pts files in directory and saves processed points to a JSON file."""
    result = {}
    # Find all files matching kk*.pts
    pts_files = glob.glob(os.path.join(directory, "kk*.pts"))
    
    for file_path in pts_files:
        # Extract number from filename (e.g., 'kk674.pts' -> '674')
        file_name = os.path.basename(file_path)
        try:
            number = int(file_name.replace('kk', '').replace('.pts', ''))
            if 1 <= number <= 1100:
                # Process the file using provided functions
                points = read_pts_file(file_path)
                result[str(number)] = points
        except ValueError:
            print(f"Warning: Skipping file '{file_name}' due to invalid number format.")
            continue
    
    # Write dictionary to JSON file
    try:
        with open(output_json, 'w') as json_file:
            json.dump(result, json_file, indent=4)
        print(f"JSON file saved to '{output_json}'")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

# Example usage
directory = "data/ExtractedFromGifs/"  # Update with your directory path
output_json = "data/combinedPts.json"  # Output JSON file path
generate_json_from_pts(directory, output_json)