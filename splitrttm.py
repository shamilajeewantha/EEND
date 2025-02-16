import os

def split_rttm(reference_rttm_path, output_dir):
    """
    Splits a large RTTM file into multiple RTTM files based on recording IDs.

    Args:
        reference_rttm_path (str): Path to the large RTTM file.
        output_dir (str): Directory where individual RTTM files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rttm_dict = {}

    # Read RTTM file and group by recording ID (second column)
    with open(reference_rttm_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # Skip malformed lines
            
            file_id = parts[1]  # Recording ID
            
            if file_id not in rttm_dict:
                rttm_dict[file_id] = []
            
            rttm_dict[file_id].append(line.strip())

    # Write each recording's annotations into separate RTTM files
    for file_id, lines in rttm_dict.items():
        output_path = os.path.join(output_dir, f"{file_id}.rttm")
        with open(output_path, "w") as out_file:
            out_file.write("\n".join(lines) + "\n")
        
        print(f"Created: {output_path} ({len(lines)} entries)")

# Example Usage
reference_rttm_path = "./rttm/rttm"  # Path to your large RTTM file
output_dir = "./rttm/split_rttms"  # Folder to store the split RTTM files

split_rttm(reference_rttm_path, output_dir)
