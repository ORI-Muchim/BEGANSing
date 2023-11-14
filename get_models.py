import os
import requests

def download_file(url, path):
    response = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as file:
        file.write(response.content)

def get_model():
    base_dir = './RVC'
    model_urls = {
        'pretrained_v2/f0D48k.pth': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/f0D48k.pth',
        'pretrained_v2/f0G48k.pth': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/f0G48k.pth',
        'hubert_base.pt': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/hubert_base.pt',
        'rmvpe.pt': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/rmvpe.pt',
        'checkpoint/default/latest_D.pt': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/latest_D.pt',
        'checkpoint/default/latest_G.pt': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/latest_G.pt',
        'checkpoint/default/default_train.yml': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/default_train.yml',
        'hifi_gan/default/do_02500000': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/do_02500000',
        'hifi_gan/default/g_02500000': 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/g_02500000',
    }

    for filename, url in model_urls.items():
        file_path = os.path.join(base_dir, filename)
        if not os.path.isfile(file_path):
            print(f"Downloading {filename}...")
            download_file(url, file_path)
            print(f"Saved {filename}.\n")
        else:
            print(f'Skipping Download... {filename} exists.')

get_model()
