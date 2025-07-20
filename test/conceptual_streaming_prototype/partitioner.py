import json

def load_operator_infos(path):
    with open(path) as f:
        logs = json.load(f)
    operator_infos = []
    for entry in logs:
        if "weight_size" not in entry:
            continue
        operator_infos.append({
            "op_name": entry["op_name"],
            "weight_size": entry["weight_size"],
            "compute_latency": entry["compute_latency"],
            "io_latency": entry["io_latency"],
        })
    return operator_infos

def partition_operators(operator_infos, Mbudget, MKV):
    chunks = []
    i = 0
    while i < len(operator_infos):
        acc_weight = 0
        acc_compute = 0
        start_idx = i
        end_idx = i
        
        current_chunk_ops = []

        while i < len(operator_infos):
            op = operator_infos[i]
            
            # Look ahead to get the I/O cost of the *next* chunk's first op
            next_io = 0
            if i + 1 < len(operator_infos):
                # Find the next operator that actually has I/O
                for k in range(i + 1, len(operator_infos)):
                    if operator_infos[k]["io_latency"] > 0:
                        next_io = operator_infos[k]["io_latency"]
                        break
            
            # Estimate peak memory: current chunk weights + KV cache + next chunk's first op weight
            next_op_weight = operator_infos[i + 1]["weight_size"] if i + 1 < len(operator_infos) else 0
            Mpeak = acc_weight + op["weight_size"] + MKV + next_op_weight

            # Condition to create a new chunk
            # 1. Peak memory exceeds budget
            # 2. I/O time for the next op is greater than the accumulated compute time of the current chunk
            if Mpeak > Mbudget or (next_io > (acc_compute + op["compute_latency"]) and acc_compute > 0):
                if not current_chunk_ops:
                    # If the first op itself violates the condition, it becomes a chunk of its own
                    current_chunk_ops.append(op)
                    acc_weight += op["weight_size"]
                    acc_compute += op["compute_latency"]
                    end_idx = i
                    i += 1
                break

            current_chunk_ops.append(op)
            acc_weight += op["weight_size"]
            acc_compute += op["compute_latency"]
            end_idx = i
            i += 1

        chunks.append({
            "chunk_id": len(chunks),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "weight_size": acc_weight,
            "compute_latency": acc_compute,
            "io_latency_of_next": next_io,
            "flash_offset": -1, # Placeholder
        })

    return chunks

def save_cmt(chunks, out_path="cmt.json"):
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunk Metadata Table saved to {out_path}")

if __name__ == "__main__":
    operator_infos = load_operator_infos("merged_profile.json")
    Mbudget = 3 * 1024 * 1024  # 3MB example budget
    MKV = 512 * 1024           # 512KB example KV cache size
    
    print(f"Partitioning with Memory Budget: {Mbudget/1024/1024:.2f} MB, KV Cache: {MKV/1024/1024:.2f} MB")
    
    chunks = partition_operators(operator_infos, Mbudget, MKV)
    save_cmt(chunks)
