import os
import requests

hubert = './RVC/hubert_base.pt'

def get_model():
    if not os.path.isfile(hubert):
        url1 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/f0D48k.pth'
        url2 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/f0G48k.pth'
        url3 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/hubert_base.pt'
        url4 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/rmvpe.pt'
        url5 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/latest_D.pt'
        url6 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/latest_G.pt'
        url7 = 'https://github.com/ORI-Muchim/BEGANSing/releases/download/v1.0/default_train.yml'

        print("Downloading Pretrained Discriminator Model...")
        response1 = requests.get(url1, allow_redirects=True)

        print("Downloading Pretrained Generator Model...")
        response2 = requests.get(url2, allow_redirects=True)
        
        print("Downloading Hubert Model...")
        response3 = requests.get(url3, allow_redirects=True)
        
        print("Downloading RMVPE Model...")
        response4 = requests.get(url4, allow_redirects=True)
        
        print("Downloading BEGANSing Checkpoint...")
        response5 = requests.get(url5, allow_redirects=True)
        response6 = requests.get(url6, allow_redirects=True)
        response7 = requests.get(url7, allow_redirects=True)

        directory = './RVC/pretrained_v2'
        directory2 = './RVC'
        directory3 = './checkpoint/default'
        directory4 = 'hifi-gan/default'

        pretrained_discriminator_model = os.path.join(directory, 'f0D48k.pth')
        pretrained_generator_model = os.path.join(directory, 'f0G48k.pth')
        hubert_model = os.path.join(directory2, 'hubert_base.pt')
        rmvpe_model = os.path.join(directory2, 'rmvpe.pt')
        begansing_check_D_model = os.path.join(directory3, 'latest_D.pt')
        begansing_check_G_model = os.path.join(directory3, 'latest_G.pt')
        begansing_config = os.path.join(directory3, 'default_train.yml')
        
        with open(pretrained_discriminator_model, 'wb') as file:
            file.write(response1.content)
        print("Saving Pretrained Discriminator Model...")

        with open(pretrained_generator_model, 'wb') as file:
            file.write(response2.content)
        print("Saving Pretrained Generator Model...")
        
        with open(hubert_model, 'wb') as file:
            file.write(response3.content)
        print("Saving Hubert Model...")
        
        with open(rmvpe_model, 'wb') as file:
            file.write(response4.content)
        print("Saving RMVPE Model...\n")
        
        with open(begansing_check_D_model, 'wb') as file:
            file.write(response5.content)
        print("Saving BEGANSing Discriminator Model...")
        
        with open(begansing_check_G_model, 'wb') as file:
            file.write(response6.content)
        print("Saving BEGANSing Generator Model...")
        
        with open(begansing_config, 'wb') as file:
            file.write(response7.content)
        print("Saving BEGANSing Config File...")
    else:
        print('Skipping Download... Model exists.')
