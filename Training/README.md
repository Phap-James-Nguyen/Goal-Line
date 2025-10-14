# Object Recognition Training Directory

This directory contains all files, scripts, and results related to the object recognition training experiment conducted on the **George Mason University Hopper Cluster**.

---

## 📁 Directory Overview

### Main Components

- **`run.slurm`** — SLURM batch script used to submit the training job on the Hopper cluster.
- **`train.py`** — Python training script that handles model training, validation, and result generation.

- **Result Visualizations ( Subdirs ):**
  - `confusion_matrix.png` — Class-level confusion matrix.
  - `confusion_matrix_normalized.png` — Normalized confusion matrix.
  - `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png` — Performance metrics curves.
  - `results.csv` and `results.png` — Numerical and graphical summaries of training performance.
  - `labels.jpg`, `labels_correlogram.jpg` — Label distribution and class correlation visuals.
- **Validation Samples:**
  - `val_batch*_labels.jpg` — Ground truth annotations for validation batches.
  - `val_batch*_pred.jpg` — Model predictions for validation batches.
- **Training Samples:**
  - `train_batch*.jpg` — Training sample visualizations.

---

## 🧠 Training Script — `train.py`

The `train.py` script handles:
 - Loads a pre-trained YOLOv8 model for transfer learning.
 - Trains on the dataset specified in data.yaml.
 - Resizes input images to 640×640 resolution.
 - Uses a batch size of 16 and runs up to 50 epochs.
 - Applies early stopping if validation performance stalls for 7 epochs.
 - Uses SGD optimizer with:
 - Initial learning rate: 0.001
 - Weight decay: 0.0005
 - Runs validation during training (val=True).
 - Treats all detections as one class (single_cls=True).

It can be run directly using:
```bash
python train.py 
```

---

## 🧾 SLURM Job Script — `run.slurm`

The `run.slurm` file automates training on the Hopper cluster. It requests computing resources, loads necessary modules, and executes the training script.

Example job submission command:
```bash
sbatch run.slurm
```

### To check job progress:
```bash
squeue -u $USER
```

### To cancel a running job:
```bash
scancel <job_id>
```

---

## 📊 Output Summary

After training completes, this directory contains all plots, logs, and metrics for analysis. The best model weights can be found in the `weights/` subdirectory.

