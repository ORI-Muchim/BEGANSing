import os
import glob
import sys
from ruamel.yaml import YAML

def update_text_file_in_yaml(yaml_path):
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        current_text_file_path = data.get('text_file')
        if current_text_file_path is None:
            print('text_file key not found in YAML file.')
            return

        directory, file_name_with_extension = os.path.split(current_text_file_path)
        file_name, extension = os.path.splitext(file_name_with_extension)

        print('Current text file base name:', file_name)
        
        new_file_base_name = sys.argv[2]
        new_text_file_path = os.path.join(directory, new_file_base_name + extension)
        data['text_file'] = new_text_file_path
        
        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file)
        
        print('text_file has been updated to:', new_text_file_path)
        
    except Exception as e:
        print('Error:', str(e))


def find_index_files(directory):

    pattern = os.path.join(directory, 'added*.index')
    files = glob.glob(pattern)
    
    return files
