#!/usr/bin/env python3
"""
Update all JSON prompt files to include missing hyperparameters.
"""

import json
import os
import glob

def update_json_file(file_path):
    """Update a single JSON file to include missing hyperparameters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def update_item(item):
            # Add missing fields with defaults
            if 'top_p' not in item:
                item['top_p'] = 0.9
            if 'repetition_penalty' not in item:
                item['repetition_penalty'] = 1.0
            if 'enable_repetition_penalty' not in item:
                item['enable_repetition_penalty'] = False
            return item
        
        # Update based on structure
        if isinstance(data, list):
            # Array of prompts
            for item in data:
                update_item(item)
        else:
            # Single prompt
            update_item(data)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated: {file_path}")
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def main():
    # Find all JSON files in the prompt directory
    prompt_dir = "./prompt"
    json_files = glob.glob(os.path.join(prompt_dir, "*.json"))
    
    if not json_files:
        print("No JSON files found in prompt directory")
        return
    
    print(f"Found {len(json_files)} JSON files to update:")
    for file_path in json_files:
        update_json_file(file_path)
    
    print("All JSON files updated successfully!")

if __name__ == "__main__":
    main()
