# BEGANSing + RVC + AudioSuperResolution

## Korean Singing Voice Synthesis + Singing Voice Conversion(SVS + SVC)

The system generates singing voice from a given text and MIDI in an end-to-end manner. 


![model architecture final 2 3](https://user-images.githubusercontent.com/15067112/81911402-3917fe80-9608-11ea-9718-8a61b564a618.jpg)
<p align="center"><b>Overview of the proposed system</b></p>

# Contents
- [Installation](#installation)
- [Prepare Datasets](#prepare-dataset)
- [Configuration](#configuration)
- [Preprocessing & Training](#preprocessing--training)
- [Usage](#usage)
- [Results](#results)
- [To-Do](#to-do)
- [References](#references)

## Installation
- A Windows/Linux system with a minimum of `16GB` RAM.
- A GPU with at least `12GB` of VRAM.
- Python >= 3.8
- Anaconda installed.
- Pytorch installed.
- CUDA 11.7 installed.

Pytorch install command:
```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

CUDA 11.8 install:
```sh
https://developer.nvidia.com/cuda-11-8-0-download-archive
```

---

1. **Create an Anaconda environment:**

```sh
conda create -n begansing python=3.9
```

2. **Activate the environment:**

```sh
conda activate begansing
```

3. **Clone this repository to your local machine:**

```sh
git clone https://github.com/ORI-Muchim/BEGANSing.git
```

4. **Navigate to the cloned directory:**

```sh
cd BEGANSing
```

5. **Install the necessary dependencies:**

```sh
pip install -r requirements.txt
```

# Prepare Dataset
Inside the cloned folder, there is a folder called `./test_datasets`. You can put the MIDI file and text file in it according to the format. MIDI and text should be arranged in the same number unconditionally. As an example, I will provide GFRIEND's "Rough" MIDI and text. And for the dataset to change the voice from the generated vocals, you can create a folder with the speaker's name in the `./datasets` folder and put voice data for Retrieval Voice Conversion (RVC) in it. The following shows the ./datasets format.

```
BEGANSing
├────datasets
│       ├───kss
│       │   ├────1_0000.wav
│       │   ├────1_0001.wav
│       │   └────...
│       ├───{speaker_name}
│       │    ├───1.wav
└───────└────└───2.wav
```
This is just an example, and it's okay to add more speakers.

# Preprocessing & Training

This pre-trained model is a model in which an additional 100 epochs was trained. For Preprocessing and Training, see [Preprocessing](https://github.com/SoonbeomChoi/BEGANSing#preprocessing), [Training](https://github.com/SoonbeomChoi/BEGANSing#training) in the original repository.


# Usage
```bash
python main.py {speaker_name} {song} {pitch_shift} --audiosr
```

If the speaker is male, it is recommended to set the {pitch_shift} value to `-12`, and if she is female, set it to `0`.

The `--audiosr` option up-samples a voice generated at 22050hz to 48000hz. Use this option for those who have excellent graphics cards or don't mind taking a long time to generate a voice, or remove it if not.

# Results
Audio samples at: https://soonbeomchoi.github.io/saebyulgan-blog/. Model was trained at RTX3090 24GB with batch size 32 for 2 days.
![BEGANSing tensorboard](https://user-images.githubusercontent.com/15067112/82179828-34aa5900-991a-11ea-9f73-b85aad109cea.png)

# To-Do
- Vocoder Change(Griffin-Lim -> HiFi-GAN) - Work In Progress!

# References
- g2p/korean_g2p.py from https://github.com/scarletcho/KoG2P
- utils/midi_utils.py from Madmom, https://madmom.readthedocs.io/en/latest/
