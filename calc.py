import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def load_rttm_folder(folder_path):
    """Load all RTTM files from a folder and return a list of pyannote.core.Annotation."""
    annotations = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.rttm'):
            file_path = os.path.join(folder_path, filename)
            annotations.append(load_rttm(file_path))
    return annotations

def load_rttm(file_path):
    """Load RTTM file and return pyannote.core.Annotation."""
    annotation = Annotation(uri=os.path.basename(file_path))
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            onset = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(onset, onset + duration)] = speaker
    return annotation

def calculate_der(reference_folder, hypothesis_folder):
    # Construct full paths to reference and hypothesis folders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reference_folder = os.path.join(script_dir, reference_folder)
    hypothesis_folder = os.path.join(script_dir, hypothesis_folder)
    
    # Load reference and hypothesis RTTM files from folders
    reference_annotations = load_rttm_folder(reference_folder)
    hypothesis_annotations = load_rttm_folder(hypothesis_folder)
    
    # Initialize DiarizationErrorRate metric
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

    # Compute DER for each pair of reference and hypothesis annotations
    total_der = 0.0
    total_confusion = 0.0
    total_false_alarm = 0.0
    total_missed_detection = 0.0

    for ref_ann, hyp_ann in zip(reference_annotations, hypothesis_annotations):
        detailed_report = metric(ref_ann, hyp_ann, detailed=True)
        
        # Print the detailed report for each file pair
        print(detailed_report)
        
        # Accumulate the total values for each metric
        total_der += detailed_report['diarization error rate']
        total_confusion += detailed_report['confusion']
        total_false_alarm += detailed_report['false alarm']
        total_missed_detection += detailed_report['missed detection']

    # Average DER over all pairs
    if len(reference_annotations) > 0:
        average_der = total_der / len(reference_annotations)
    else:
        average_der = 0.0

    if len(reference_annotations) > 0:
        average_confusion = total_confusion / len(reference_annotations)
    else:
        average_confusion = 0.0
    
    if len(reference_annotations) > 0:
        average_false_alarm = total_false_alarm / len(reference_annotations)
    else:
        average_false_alarm = 0.0

    if len(reference_annotations) > 0:
        average_missed_detection = total_missed_detection / len(reference_annotations)
    else:
        average_missed_detection = 0.0
    
    return average_der, average_confusion, average_false_alarm, average_missed_detection

# Folder names containing RTTM files (in the same directory as this script)
# pyannote as refernce
reference_folder = './rttm/manual_rttms'
# reference_folder = '/content/drive/MyDrive/speaker_diarization/EEND_output/split_rttms'
# hypothesis_folder = 'diaper_annotations'
# hypothesis_folder = 'powerset_annotations'
# hypothesis_folder = '/content/drive/MyDrive/speaker_diarization/EEND_output/hypothesis_rttms/rttms'
hypothesis_folder = './rttm/hypothesis_rttms/ep100'



# Calculate average DER
average_der, average_confusion, average_false_alarm, average_missed_detection = calculate_der(reference_folder, hypothesis_folder)
print(f'hypothesis_folder: {hypothesis_folder}')
print(f'Average Diarization Error Rate (DER): {average_der}')
print(f'Average Confusion: {average_confusion}')
print(f'Average False Alarm: {average_false_alarm}')
print(f'Average Missed Detection: {average_missed_detection}')