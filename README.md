# Using Machine Learning to Detect Deepfake Artifacts

This repository contains the Google Colab notebooks, fixed data manifests, reference results, plots, and supporting files used for an MSc Artificial Intelligence dissertation investigating within-dataset and cross-dataset deepfake detection.

The experiments compare face-focused preprocessing, transfer learning, fine-tuning, Self-Blended Images (SBI), optimisation and model-capacity configurations, and frequency-informed approaches. Models were trained using FaceForensics++ data and evaluated internally and on a fixed Celeb-DF v2 pilot.

## Repository Structure

```text
deepfake-detection-dissertation/
├── notebooks/       Google Colab experiment notebooks
├── manifests/       Fixed FF++ split and Celeb-DF pilot manifests
├── checkpoints/     Checkpoint download instructions
├── results/         Reference experimental results
├── plots/           Reference plots and figures
├── requirements.txt
├── .gitignore
└── README.md
```

Datasets and trained model files are not stored directly in this repository because of their size. The notebooks download the required public datasets during execution. Pre-trained checkpoints are provided separately through OneDrive.

## Main Software Environment

The experiments were developed for Google Colab with Google Drive mounted for persistent storage.

The main project directory used by the notebooks is:

```text
/content/drive/MyDrive/deepfake_project/
```

A CUDA-enabled Colab runtime is recommended for model training. Data preparation and evaluation may also run on CPU, although execution will be slower.

## Checkpoints

Pre-trained model checkpoints are stored separately in OneDrive.

See:

```text
checkpoints/README.md
```

Download the checkpoints and copy them into:

```text
/content/drive/MyDrive/deepfake_project/saved_models/
```

The checkpoint filenames must not be changed because the evaluation notebooks expect the original names.

## Datasets

The notebooks download the required datasets through KaggleHub during execution:

* FaceForensics++ source: `hungle3401/faceforensics`

* Celeb-DF v2 source: `reubensuju/celeb-df-v2`

The full datasets do not need to be downloaded manually or added to this GitHub repository.

The first download may require the user to accept the relevant Kaggle dataset conditions or authenticate with Kaggle, depending on the current Kaggle access requirements.

## Fixed Manifests

Two fixed manifests are included:

```text
manifests/ffpp_video_split_manifest.json
manifests/celebdf_pilot_manifest.json
```

They preserve the exact source-video split and external pilot selection used in the dissertation.

Notebook `00_Project_Setup_and_FFPP_Data_Preparation.ipynb` automatically
downloads both manifests to the required Google Drive locations when they are
not already present. Existing manifest files are preserved.

The manifests ensure that the same FaceForensics++ video split and Celeb-DF v2
pilot selection are used on every run.

Using these files prevents the notebooks from generating a different split or pilot selection.

## Recommended Reproduction Workflow

The recommended approach uses the supplied checkpoints. Retraining is optional and is not required to reproduce the reported results.

### Opening Notebooks in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab.
3. Enter this repository address:

```text
https://github.com/sawket/deepfake-detection-dissertation
```

4. Select the required notebook from the displayed list.
5. Connect to Google Drive when prompted.

Follow these steps whenever the instructions below ask you to open a notebook.

### Route 1: Reproduce All Results Using Checkpoints

1. Open `00_Project_Setup_and_FFPP_Data_Preparation.ipynb` and use:

   ```python
   INSTALL_DEPENDENCIES = True
   PREPARE_FFPP_DATA = True
   ```

   Run all cells. This creates the project folders, downloads the fixed manifests and prepares the FaceForensics++ dataset.

2. Open and run:

   ```text
   02a_Face_Preprocessing.ipynb
   ```

   This creates the 224 × 224 FaceForensics++ face crops.

3. Open and run:

   ```text
   05_CelebDF_Pilot_Preparation.ipynb
   ```

   This prepares the fixed Celeb-DF v2 pilot using full frames and 224 × 224 face crops.

4. Open and run:

   ```text
   02c_B4_Test_Face_Preparation.ipynb
   ```

   This creates the 380 × 380 FaceForensics++ and Celeb-DF face crops required by the EfficientNet-B4 configurations.

5. Follow the instructions in `checkpoints/README.md` to download the supplied checkpoints. Place them in:

   ```text
   /content/drive/MyDrive/deepfake_project/saved_models/
   ```

6. Run the `04` evaluation notebooks to reproduce the FaceForensics++ results.


7. Run the `05a`--`05k` evaluation notebooks to reproduce the Celeb-DF v2 pilot results.

Notebook `02b_SBI_Preprocessing.ipynb` and the `03` training notebooks are required only for optional model retraining.
### Route 2: Run the Final E3b Demonstrator

1. Open `00_Project_Setup_and_FFPP_Data_Preparation.ipynb`.

2. Set:

```python
INSTALL_DEPENDENCIES = True
PREPARE_FFPP_DATA = False
```

Run all cells. This creates the required folders and installs the dependencies without preparing the datasets.

3. Download:

```text
efficientnet_b0_official_sbi_best.pth
```

Place it in:

```text
/content/drive/MyDrive/deepfake_project/saved_models/
```

4. Open and run:

```text
optional/06b_E3b_Gradio_Application.ipynb
```

This route does not require dataset preparation or model retraining.

## Optional Full Retraining

The `02b` and `03` notebooks document the preprocessing and model training used in the dissertation. Run these notebooks only if complete retraining is required. Retraining requires additional time, Google Drive storage and GPU resources.

## Experiment Mapping

| Experiment | Description                                        |
| ---------- | -------------------------------------------------- |
| E1         | Shallow CNN using full frames                      |
| E1.1       | Shallow CNN using face crops                       |
| E2         | Frozen EfficientNet-B0 using full frames           |
| E2.1       | Fully fine-tuned EfficientNet-B0 using full frames |
| E2.2       | Fully fine-tuned EfficientNet-B0 using face crops  |
| E3a        | Adapted Self-Blended Images                        |
| E3b        | Official SBI procedure with EfficientNet-B0        |
| E3c        | Official SBI optimisation configuration with SAM   |
| E3d        | Official SBI procedure with EfficientNet-B4        |
| E4         | FSBI-DWT frequency-informed configuration          |
| E5         | FreqBlender SBI configuration with EfficientNet-B4 |

## Decision Threshold and Video Aggregation

The evaluation notebooks use the predefined decision threshold of:

```text
0.5
```

The Celeb-DF v2 pilot is treated as an external target test set and must not be used to optimise or retrospectively select a different threshold.

Frame-level manipulation scores are aggregated into video-level scores using mean averaging across the ten sampled frames belonging to each video.

## Dependencies

The main dependencies are listed in:

```text
requirements.txt
```

In Google Colab, the notebooks install their required packages when their dependency-installation control is enabled.

The `torch` and `torchvision` packages are not restricted to CPU-only builds in `requirements.txt`, allowing Colab to use a compatible GPU-enabled installation when a GPU runtime is available.

## Reference Results and Plots

The `results` and `plots` directories contain the saved outputs from the principal experimental runs reported in the dissertation.

These files are included for verification and comparison. Rerunning an evaluation notebook writes runtime outputs to the corresponding Google Drive project directories and does not modify the reference files stored in the GitHub repository.

## Reproducibility Notes

For the closest reproduction:

* use the supplied manifests;

* retain the exact checkpoint filenames;

* preserve the class mappings defined in each notebook;

* preserve each checkpoint’s required preprocessing;

* keep the external decision threshold fixed at `0.5`;

* do not rebuild the Celeb-DF pilot;

* do not mix frames from different dataset splits;

* record the Colab hardware and installed package versions used for any new run.

## Limitations

The external evaluation is based on a fixed pilot containing 40 Celeb-DF v2 videos rather than the complete dataset. Each experiment is represented by one principal training run. The repository therefore supports reproduction of the implemented experimental workflow and reported principal results, but it does not establish deployment-level performance.

## Author

Sawket Ahmed

MSc Artificial Intelligence

Manchester Metropolitan University

