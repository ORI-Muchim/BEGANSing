import os
import sys
import shutil
from main_util import update_text_file_in_yaml, find_index_files

# Init
model_name = sys.argv[1]
song_name = sys.argv[2]
f0_up_key = int(sys.argv[3])  # transpose value
input_path = f"../samples/latest_G_{song_name}.wav"
output_path = f"../samples/latest_G_{song_name}.wav"
model_path = f"./weights/{model_name}.pth"
device = "cuda:0"
f0_method = "rmvpe"  # pm or harvest or crepe

yaml_path = "./config/default_infer.yml"
update_text_file_in_yaml(yaml_path)

# BEGANSing Inference
os.system(f"python infer.py -c config/default_train.yml config/default_infer.yml --device 0")

# RVC Inference
os.chdir("./RVC")
    
# Assuming model_name is initialized somewhere before this code
model_file = f"./weights/{model_name}.pth"

if not os.path.isfile(model_file):
    os.system(f"python oneclickprocess.py --name {model_name}")
else:
    print('Skipping Training... Model already exists.')
    
index_directory = f'./logs/{model_name}'
file_index_list = find_index_files(index_directory)
if file_index_list:
    file_index = file_index_list[0]

os.system(f"python .\infer_cli.py {f0_up_key} {input_path} {output_path} {model_path} {file_index} {device} {f0_method}")

# Adjust SR
temp_output_path = f"{output_path}_temp.wav"
os.system(f"ffmpeg -y -i {output_path} -ar 22050 -ac 1 {temp_output_path}")
shutil.move(temp_output_path, output_path)

# AudioSR-Upsampling
os.chdir("../AudioSR-Upsampling")
os.system(f"python main.py")

print("All Process Finished.")
