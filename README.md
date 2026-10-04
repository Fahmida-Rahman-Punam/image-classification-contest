

This repository contains the complete inference pipeline and prediction output for the NeuroLab Image Classification contest.

## 📌 Overview
* **Dataset Size**: 1,070 test images (`.jpg`, `.png`, `.jpeg`)
* **Framework**: TensorFlow / Keras, Pandas, NumPy
* **Objective**: Predict numeric class targets (0–9) for test images and output ordered predictions.

## 📁 Repository Structure
* `Untitled1.ipynb` — Full end-to-end Python notebook containing model setup, path collection using `glob`, pre-processing, and inference loops.
* `submission.csv` — Final generated predictions containing ordered `id` and predicted `target` columns for all 1,070 test images.

## 🛠️ Pipeline Details
1. **Data Collection**: Uses `glob.glob` to recursively retrieve all test image paths from Google Drive.
2. **Preprocessing**: Resizes images to `128x128` pixels and prepares batches using `tf.keras.utils`.
3. **Inference & Post-processing**: Runs model predictions, extracts top class indices via `np.argmax`, sorts by numeric ID, and exports to CSV format.
