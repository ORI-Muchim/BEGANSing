import os
import random

target_directory = './wav'
wav_files = [file for file in os.listdir(target_directory) if file.endswith('.wav')]

with open('train_list.txt', 'w', encoding='utf-8') as f:
    for wav_file in wav_files:
        file_name_without_extension = os.path.splitext(wav_file)[0]
        f.write(file_name_without_extension + '\n')

print("train_list.txt is saved.")

val_file = int(len(wav_files) * 0.01)
selected_files = random.sample(wav_files, val_file)

with open('valid_list.txt', 'w', encoding='utf-8') as f:
    for wav_file in selected_files:
        file_name_without_extension = os.path.splitext(wav_file)[0]
        f.write(file_name_without_extension + '\n')

print("valid_list.txt is saved.")
