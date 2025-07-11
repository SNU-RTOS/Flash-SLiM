import pandas as pd
import re
import sys

def parse_tensor_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        match = re.match(r'\s*(\d+)\s+(0x[0-9a-fA-F]+)\s+([0-9\.]+ MB)\s+(\w+)\s+(\d+)\s+(\S+)\s+Nodes: (\d+)', line)
        if match:
            data.append(match.groups())
    
    df = pd.DataFrame(data, columns=["Tensor ID", "Address", "Size", "Data Type", "Usage Count", "Shared With", "Used By Nodes"])
    return df

if __name__ == "__main__":
    
    if sys.argc < 2:
        print("python3 convert_csv <inputfile> ")
        quit()
    input_file = sys.argv[1]  # 입력 파일 경로
    output_file = f"{input_file}_output"  # 출력 CSV 파일 경로
    
    df = parse_tensor_data(input_file)
    df.to_csv(output_file, index=False)
    print(f"CSV file saved as {output_file}")
