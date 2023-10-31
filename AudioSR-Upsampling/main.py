import os
import subprocess
import json
import tempfile
from audiosr import build_model, super_resolution, save_wave

def get_sample_rate(file_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate',
        '-of', 'json',
        file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = json.loads(result.stdout)
    return int(output['streams'][0]['sample_rate'])

def get_duration(file_location):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'format=duration',
        '-sexagesimal',
        '-of', 'json',
        file_location
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = json.loads(result.stdout)
    return output['format']['duration']

def remove_silence(duration, input_path, output_path):
    command = [
        'ffmpeg',
        '-y',
        '-ss', '00:00:00',
        '-i', input_path,
        '-t', duration,
        '-c', 'copy',
        output_path
    ]
    subprocess.run(command)
    os.remove(input_path)

tmp_dir = tempfile.gettempdir()
datasets_path = '../samples'
wav_files = [f for f in os.listdir(datasets_path) if f.endswith('.wav')]
audiosr = build_model(model_name='basic', device="auto")

for wav_file in wav_files:
    input_path = os.path.join(datasets_path, wav_file)
    duration = get_duration(input_path)
    try:
        waveform = super_resolution(
            audiosr,
            input_path,
            seed=42,
            guidance_scale=3.5,
            ddim_steps=50,
            latent_t_per_second=12.8
        )
        base_name = os.path.splitext(wav_file)[0]
        tmp_file_path = os.path.join(tmp_dir, f"{base_name}.wav")
        save_wave(waveform, tmp_dir, name=base_name, samplerate=48000)
        remove_silence(duration, tmp_file_path, input_path)
    except Exception as e:
        print(f"An error occurred processing {wav_file}: {e}")
