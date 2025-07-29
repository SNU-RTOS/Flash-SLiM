#!/usr/bin/env python3
"""
JSON prompt parser for text generation script.
Handles both single prompts and arrays of prompts.
Supports both string and array formats for the prompt field.
"""

import json
import sys
import os

def parse_prompt_field(prompt_field):
    """Parse prompt field which can be either a string or array of strings."""
    if isinstance(prompt_field, str):
        return prompt_field.strip()
    elif isinstance(prompt_field, list):
        # Join array elements with newlines
        return '\n'.join(str(line) for line in prompt_field).strip()
    else:
        return str(prompt_field).strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 parse_json_prompt.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"ERROR: File not found: {json_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Array of prompts
            print(f"ARRAY {len(data)}")
            for i, item in enumerate(data):
                tokens = item.get('tokens', 0)
                prompt = parse_prompt_field(item.get('prompt', ''))
                temperature = item.get('temperature', 0.7)
                top_k = item.get('top_k', 40)
                top_p = item.get('top_p', 0.9)
                repetition_penalty = item.get('repetition_penalty', 1.0)
                enable_repetition_penalty = item.get('enable_repetition_penalty', False)
                
                print(f"ITEM {i} {tokens} {temperature} {top_k} {top_p} {repetition_penalty} {enable_repetition_penalty}")
                print(f"PROMPT_START")
                print(prompt)
                print(f"PROMPT_END")
        else:
            # Single prompt object
            tokens = data.get('tokens', 0)
            prompt = parse_prompt_field(data.get('prompt', ''))
            temperature = data.get('temperature', 0.7)
            top_k = data.get('top_k', 40)
            top_p = data.get('top_p', 0.9)
            repetition_penalty = data.get('repetition_penalty', 1.0)
            enable_repetition_penalty = data.get('enable_repetition_penalty', False)
            
            print(f"SINGLE {tokens} {temperature} {top_k} {top_p} {repetition_penalty} {enable_repetition_penalty}")
            print(f"PROMPT_START")
            print(prompt)
            print(f"PROMPT_END")
            
    except json.JSONDecodeError as e:
        print(f"JSON_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
