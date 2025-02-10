import os
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

def load_rttm(filepath):
    annotation = Annotation()
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_id, start_time, duration, speaker = parts[1], float(parts[3]), float(parts[4]), parts[7]
            annotation[Segment(start_time, start_time + duration)] = speaker
    return annotation

# Paths
hypothesis_rttm_dir = "/content/EEND/output_directory/hypothesis_rttms/epochs20-24/timeshuffleTrue/spk_qty2_spk_qty_thr-1.0/detection_thr0.5/median11/rttms"
reference_rttm_path = "/content/Speaker-ID-Cluster/EEND/egs/mini_librispeech/v1/data/simu/data/dev_clean_2_ns2_beta2_500/rttm"
output_file_path = "der_results.txt"

# Initialize DER metric
metric = DiarizationErrorRate()

# Load reference annotation
reference_annotation = load_rttm(reference_rttm_path)

# Process each hypothesis RTTM file
total_der = 0.0
num_files = 0

# Write results to a text file
with open(output_file_path, "w") as output_file:
    output_file.write("DER for each file:\n")
    print("DER for each file:")

    for filename in os.listdir(hypothesis_rttm_dir):
        if filename.endswith(".rttm"):
            hypothesis_rttm_path = os.path.join(hypothesis_rttm_dir, filename)
            hypothesis_annotation = load_rttm(hypothesis_rttm_path)
            
            # Compute DER
            der = metric(reference_annotation, hypothesis_annotation)
            result_line = f"{filename}: {100 * der:.2f}%\n"
            
            # Write and print results
            output_file.write(result_line)
            print(result_line.strip())
            
            total_der += der
            num_files += 1

    # Calculate and display the average DER
    if num_files > 0:
        avg_der = (total_der / num_files) * 100
        output_file.write(f"\nTotal Average DER: {avg_der:.2f}%\n")
        print(f"\nTotal Average DER: {avg_der:.2f}%")
    else:
        output_file.write("No RTTM files found.\n")
        print("No RTTM files found.")
