# VL-FOG: Vision–Language Flow Gating Framework for Early Accident Anticipation - [TAISC 2025](https://sites.google.com/view/avss2025-tw/taisc)

This repository contains the main training pipeline for the **VL-FOG** model, designed for the TAISC (Taiwan Accident Classification) Challenge. The model predicts imminent traffic accidents from dash-cam video using a vision-language approach with vehicle detection via YOLO.



| #   | Participant | Entries | Date              | Score   | Accuracy | F1 Score | AUC  |
|-----|-------------|---------|-------------------|---------|----------|----------|------|
| 🥇  | **kane110 (Our team)** | **1**     | **2025-07-11 02:25**  | **0.67** | **0.70**     | **0.63**     | **0.73** |
| 🥈  | qyming      | 1       | 2025-07-12 22:43  | **0.66** | 0.72     | 0.60     | 0.71 |
| 🥉  | hsu_lab     | 1       | 2025-07-13 06:49  | **0.65** | 0.69     | 0.60     | 0.71 |


## Table of Contents

| Guide            | Description                                  |
|------------------|----------------------------------------------|
| Test Guide | Generate Submission File (see below)         |
| [Training Guide](training_guide.md)                | Full training instructions (external guide)  |

---

## 🚀 Quick Start for Make Submission File

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Prepare Data**
- Download the [offical AVA dataset](https://drive.google.com/file/d/1F1Vaat_ZrpITtDtlo4WB5cwUHEfMYon2/view)
- Place your training CSVs and frame folders as specified in your `config/training_config_stepLR.yaml` ([see project structure](#project-structure)).

```
data/AVA_Dataset/
├── freeway_train.csv
├── road_train.csv
├── freeway/train/<video_folder>/00001.jpg ...
└── road/train/<video_folder>/00001.jpg ...
```

### 3. **Download YOLO Weights**
- Download the [YOLO Weights](https://drive.google.com/file/d/1wC4Wxtlg7bv4W7gbrsHd2lcYh59Ow9me/view?usp=sharing) (e.g., `yolo11x_trained.pt`).

Or run command
```bash
gdown 1wC4Wxtlg7bv4W7gbrsHd2lcYh59Ow9me
```
- Place the weights file in your project directory or a path of your choice.
- In `utils/extract_bb.py`, set the `YOLO_WEIGHTS` variable to the path of your weights file:
  ```python
  YOLO_WEIGHTS = "yolo11x_trained.pt"
  ```

### 4. **Extract Vehicle Bounding Boxes**

- Run the bounding box extraction script for the desired split:
  ```bash
  python utils/extract_bb.py --split test  # For testing - generate submission file
  ```
- This will use YOLO to detect vehicles in all frames and cache the results.
- **Bounding boxes are saved as compressed pickle files** in `.bb_cache` folders inside each video directory (e.g., `data/AVA_Dataset/freeway/test/.bb_cache/`).
- You only need to run this step once per dataset (unless you change the frames or YOLO weights).

### 5. **Edit Config** & **Generate Prediction File**

- **Model Weight:** Download the [Pretrained Model](https://drive.google.com/file/d/1mfk1D4iDenlneQmqPAYGWWmit6xO8xKs/view?usp=sharing) and place it at the path specified in your config (e.g., `models/best_model.pth`).  

```bash
gdown 1mfk1D4iDenlneQmqPAYGWWmit6xO8xKs
```
- **Important:** Before running `test.py`, set the correct `model_path`, the dataset path (`frame_root`), and the `output_csv` file in your config file (e.g., `config/training_config_stepLR.yaml`):
  ```yaml
  test:
    model_path: "models/best_model.pth"
    frame_root: 
      freeway: data/AVA_Dataset/freeway/test
      road: data/AVA_Dataset/road/test
    output_csv: "results/test_predictions.csv"
  ```- Then run:
  ```bash
  python test.py
  ```
- The prediction file will be saved to the path specified in your config.


## **Note** 
Training on different devices may yield different results, even with the same configuration. Additionally, the results are also affected by the object detection performance from YOLO. The results presented here were obtained under optimal conditions during our best training runs.


## 🧪 Data Splitting: train_df.csv and val_df.csv

After extensive experiments with **k-fold cross-validation (k=5)**, we found that a particular split of the data led to the best model performance. As a result, we use `data/train_df.csv` and `data/val_df.csv` as the fixed training and validation sets for all experiments. This split consistently yields better results than random or stratified splits.

- `data/train_df.csv`: CSV listing the best training set (file_name, risk, etc.)
- `data/val_df.csv`: CSV listing the best validation set

If you want to reproduce the better results, use these files for training and validation.

---

<details>
<summary id="project-structure">Project Structure</summary>

```
TAISC-Challenge/
├── main.py                # Main training script
├── test.py                # Model evaluation script
├── launch_board.py        # Launch TensorBoard easily
├── train.sh               # Shell script for training
├── config/
│   ├── training_config_stepLR.yaml # Main configuration file
│   └── training_config_plateau.yaml
├── data/
│   ├── data_loader.py     # Data loading utilities
│   ├── dataset.py         # Dataset class with caching
│   ├── train_df.csv       # Best training split
│   ├── val_df.csv         # Best validation split 
│   └── augmentation.py    # Data augmentation
│   └── AVA_Dataset/       # <--- Folder containing video frame folders 
│       ├── freeway_train.csv
│       ├── road_train.csv
│       ├── freeway/
│       │   └── train/
│       │       └── <video_folder>/
│       │           ├── 00001.jpg
│       │           ├── 00002.jpg
│       │           └── ...
│       └── road/
│           └── train/
│               └── <video_folder>/
│                   ├── 00001.jpg
│                   ├── 00002.jpg
│                   └── ...
├── models/
│   ├── model_loader.py    # Model loading logic
│   └── clip_fusion_net.py # Model architecture
├── utils/
│   ├── config.py          # Config loading
│   ├── setup.py           # Training setup utilities
│   ├── log.py             # Logging helpers
│   ├── training.py        # Training loop and helpers
│   ├── board.py           # TensorBoard utilities
│   ├── seed.py            # Random seed setup
│   ├── extract_bb.py      # YOLO vehicle detection
│   └── optical_flow.py    # Optical flow feature extraction
├── runs/                  # Training outputs (created automatically)
│   └── YYYYMMDD_HHMMSS/
│       ├── best_model.pth
│       ├── log.txt
│       └── tensorboard/
└── requirements.txt       # Python dependencies
```

</details>

---

## 🔗 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [CLIP Model](https://github.com/openai/CLIP)
- [TAISC 2025](https://sites.google.com/view/avss2025-tw/taisc)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





