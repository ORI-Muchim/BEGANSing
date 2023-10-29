import os
import subprocess
import re

def get_max_volume(input_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-af', 'volumedetect',
        '-f', 'null', '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    max_volume_match = re.search(r"max_volume: (-?\d+\.\d+) dB", result.stderr)
    
    if max_volume_match:
        return float(max_volume_match.group(1))
    return None

def convert_sample_rate_with_ffmpeg(directory, target_sample_rate=22050):
    output_directory = os.path.join(directory, "converted_files")
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            max_volume = get_max_volume(input_path)
            if max_volume is not None:
                volume_adjust = 0 - max_volume
            else:
                volume_adjust = 0

            command = [
                'ffmpeg',
                '-i', input_path,
                '-ac', '1',
                '-ar', str(target_sample_rate),
                '-af', f'volume={volume_adjust}dB',
                output_path
            ]
            subprocess.run(command)
            print(f"Converted: {filename}")

convert_sample_rate_with_ffmpeg('./')
