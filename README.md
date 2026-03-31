# Moving-CLAP

### An extension of CLAP for representing moving sound sources in a joint audio-language embedding space

# Training Method

## Step 1. Dataset Preparation
Captions for AudioCaps and pre-trained weights for the content encoder (PretrainedSED for BEATs) are included as submodules. To initialize them, run the following command:

```bash
git submodule update --init --recursive
cd data
python3 remove_cr.py
```

Place the wav files in `data/wav`.
You can find the download request link on the AudioCaps GitHub page([here](https://github.com/cdjkim/audiocaps/tree/master)).

For event labels used in pre-training, download the labels from the AudioSet([here](https://research.google.com/audioset/download.html)) page and place them under `data/audioset` as follows:

```
data
└── audioset
    ├── balanced_train_segments.csv
    ├── eval_segments.csv
    └── unbalanced_train_segments.csv
```

Then, generate the tag data:

```bash
cd data/event_label
python3 get_info.py
python3 convert_to_tag.py
```

## Step 2. Simulate Moving Sound Sources
Generate simulated datasets with moving sound sources:

```bash
bash create_data.sh
```

## Step 3. Pre-training the Spatial Information Encoder
Before training, set the absolute resource path in `PretrainedSED/config.py`. Replace `RESOURCES_FOLDER = "resources"` with the following code:

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_FOLDER = os.path.join(BASE_DIR, "resources")
```

Then, Pre-train the spatial information encoder using the Sound Event Localization and Detection (SELD) task:

```bash
cd pretrain_spatial_encoder
python3 train_spatial_encoder.py
```

## Step 4. Training CLAP
Train the proposed spatial-aware CLAP model using the following command:

```bash
bash train_moving_clap.sh
```

# How to Load Pre-trained Models
Refer to `model_usage.py` for examples on how to load the model and compute embeddings for audio and text.

# Citation
If you use Moving-CLAP in your research, please cite the following paper:

```
Not yet available.
```