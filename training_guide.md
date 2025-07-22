# Training Guide for VL-FOG: Vision–Language Flow Gating Framework for Early Accident Anticipation - [TAISC 2025](https://sites.google.com/view/avss2025-tw/taisc)


## 📊 Performance

|              | accuracy | f1 score | AUC    | Score  |
|--------------|----------|----------|--------|--------|
| Training  | 0.7746   | 0.7573   | 0.7790 | 0.7668 |
| Private test | 0.6992   | 0.6262   | 0.7314 | 0.6691 |



## 🚀 Quick Start & Workflow for Training

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

Or run command:
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
  python utils/extract_bb.py --split train # For training
  ```
- This will use YOLO to detect vehicles in all frames and cache the results.
- **Bounding boxes are saved as compressed pickle files** in `.bb_cache` folders inside each video directory (e.g., `data/AVA_Dataset/freeway/train/.bb_cache/`).
- You only need to run this step once per dataset (unless you change the frames or YOLO weights).

### 5. **Train the Model**:

**Note:** Training on different devices may yield different results, even with the same configuration. Additionally, the results are also affected by the object detection performance from YOLO. The results presented here were obtained under optimal conditions during our best training runs.

- You can use the provided shell script for training:
  ```bash
  chmod +x train.sh
  ```
  ```bash
  ./train.sh
  ```
- Or run manually:
  ```bash
  python main.py --config config/training_config_stepLR.yaml
  ```

Training logs, best model, and TensorBoard logs will be saved in a new `runs/YYYYMMDD_HHMMSS/` directory (the latest directory).

### 6. **Monitor Training**

- Use the provided script to launch TensorBoard:
  ```bash
  python launch_board.py
  ```
- This will automatically find the latest run and open TensorBoard at the correct log directory.
- You can also specify a custom log directory with `--runs_dir`.

### 7. **Evaluate/Test & Generate Prediction File**

- **Model Weight:** Download the [Pretrained Model](https://drive.google.com/file/d/1mfk1D4iDenlneQmqPAYGWWmit6xO8xKs/view?usp=sharing) and place it at the path specified in your config (e.g., `models/best_model.pth`).  

Or run command:
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
    ...
  ```
- Then run:
  ```bash
  python test.py
  ```
- The prediction file will be saved to the path specified in your config.

## 🖥️ Device Used for Training

All experiments and benchmarks in this project were conducted on:

- **GPU:** NVIDIA RTX A6000 (49GB VRAM)


## 🧪 Data Splitting: train_df.csv and val_df.csv

After extensive experiments with **k-fold cross-validation (k=5)**, we found that a particular split of the data led to the best model performance. As a result, we use `data/train_df.csv` and `data/val_df.csv` as the fixed training and validation sets for all experiments. This split consistently yields better results than random or stratified splits.

- `data/train_df.csv`: CSV listing the best training set (file_name, risk, etc.)
- `data/val_df.csv`: CSV listing the best validation set

If you want to reproduce the best results, use these files for training and validation.

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

<details>
<summary>Script Overview</summary>

### main.py
- Loads config and sets up directories, logging, and TensorBoard
- Initializes model, optimizer, loss, scheduler, and data loaders
- Runs the training loop with early stopping and best model saving
- Logs all progress and metrics
- Supports custom config via `--config` argument

### test.py
- Loads the best model and runs inference on the test set
- Saves predictions to CSV

### utils/extract_bb.py
- Uses YOLO to detect vehicles in video frames
- Caches bounding box data for efficient processing
- Supports multiprocessing for faster extraction
- **Bounding boxes are saved in `.bb_cache` folders inside each video directory.**

### utils/optical_flow.py
- Computes optical flow features using detected vehicle bounding boxes
- Implements caching for performance optimization
- Provides both basic and bounding box-aware flow extraction

### launch_board.py
- Use this script to launch TensorBoard for the latest run or a custom log directory.
- Example:
  ```bash
  python launch_board.py --latest
  ```

</details>

---

## 🔗 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [CLIP Model](https://github.com/openai/CLIP)
- [TAISC 2025](https://sites.google.com/view/avss2025-tw/taisc)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.