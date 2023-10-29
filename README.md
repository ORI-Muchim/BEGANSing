# Korean Singing Voice Synthesis based on Auto-regressive Boundary Equilibrium GAN

## Korean singing voice synthesis system using deep neural network

The system generates singing voice from a given text and MIDI in an end-to-end manner. 


![model architecture final 2 3](https://user-images.githubusercontent.com/15067112/81911402-3917fe80-9608-11ea-9718-8a61b564a618.jpg)
<p align="center"><b>Overview of the proposed system</b></p>

# Table of Contents 
- [Prerequisites](#pre-requisites)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [References](#references)

# Pre-requisites
- Install PyTorch based on official website, https://pytorch.org/.
- PyTorch version == 1.9.0
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- Then install packages in requirements.txt.
```bash
pip install -r requirements.txt
```

# Dataset

[Children's Song Dataset](https://zenodo.org/record/4785016)

- Each song must have text(.txt), MIDI(.mid) and audio(.wav) and MIDI should be temporally aligned with audio.
- Text files assume to have the same number of syllables as MIDI notes. The text is aligned using MIDI note duration.
- Currently the system only supports Korean.

# Configuration
Check configuration files in 'config' folder.
- default_train.yml: Default configuration file for preprocess.py and train.py
- default_infer.yml: Default configuration file for infer.py

Change configurations before you run following steps and important parameters are as below.
- file_structure: File structure of dataset, 1: all the files in one folder, 2: .txt, .mid, .wav are in separated folders
- dataset_path: Path for dataset
- num_proc: The number of processes especially for preprocess.py
- use_cpu: Forcing code to use cpu and ignore 'device' parameter
- device: List of CUDA device indices (e.g. device: [0, 1] will use cuda:0 and cuda:1)
- batch_size: Training batch size
- data_mode: Dataloader mode, single: loading entire data on memory, multi: loading data with queue

# Preprocessing
```bash
python preprocess.py -c config/default_train.yml --use_cpu True
```

You can speed up preprocessing by increasing the number of processes or 'num_proc'.
You can use preprocess.py with GPU but 'num_proc' should be 1.

# Training
```bash
python train.py -c config/default_train.yml --device 0 --batch_size 32
```

If your system doesn't have enough memory, you can change 'data_mode' to 'multi' which loads data with queue.
Also type following command for tensorboard monitoring.
```bash
tensorboard --logdir=checkpoint/csd
```

# Inference
```bash
python infer.py -c config/default_train.yml config/default_infer.yml --device 0
```

Specify text file and checkpoint file in the configuration and MIDI file is assumed to have same file name as text file. Together with train configuration, infer configuration is given then the configurations are merged.
You can also use pre-trained model from https://drive.google.com/file/d/1JqCD-kxba2meSlApWr06b2nzqLv3U-j_/view?usp=sharing (edit link).

# Results
Audio samples at: https://soonbeomchoi.github.io/saebyulgan-blog/. Model was trained at RTX3090 24GB with batch size 32 for 2 days.
![BEGANSing tensorboard](https://user-images.githubusercontent.com/15067112/82179828-34aa5900-991a-11ea-9f73-b85aad109cea.png)

# References
- g2p/korean_g2p.py from https://github.com/scarletcho/KoG2P
- utils/midi_utils.py from Madmom, https://madmom.readthedocs.io/en/latest/
