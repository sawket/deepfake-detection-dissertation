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

Before preparing the datasets, copy both files into:

```text
/content/drive/MyDrive/deepfake_project/
```

The final Google Drive locations must therefore be:

```text
/content/drive/MyDrive/deepfake_project/ffpp_video_split_manifest.json
/content/drive/MyDrive/deepfake_project/celebdf_pilot_manifest.json
```

Using these files prevents the notebooks from generating a different split or pilot selection.

## Recommended Reproduction Workflow

There are two supported workflows:

1. Evaluation using the supplied checkpoints.

2. Full data preparation and model training from the beginning.

The checkpoint-based evaluation workflow is recommended when the objective is to verify the reported results without repeating all training runs.

## Step 1: Open the Repository in Google Colab

Open Google Colab:

https://colab.research.google.com/

Select the **GitHub** tab, enter the URL of this repository, and choose the required notebook.

Alternatively, download an individual notebook from the `notebooks` directory and upload it to Colab.

## Step 2: Create the Google Drive Project Structure

Open:

```text
notebooks/00_Project_Setup_and_FFPP_Data_Preparation.ipynb
```

For the first setup-only run, set:

```python
INSTALL_DEPENDENCIES = True
PREPARE_FFPP_DATA = False
```

Run all cells. This mounts Google Drive and creates the expected project directories without downloading or preparing the dataset.

## Step 3: Install the Fixed Manifests

After the project directories have been created, copy the two repository manifest files into the root of the Google Drive project directory:

```text
/content/drive/MyDrive/deepfake_project/ffpp_video_split_manifest.json
/content/drive/MyDrive/deepfake_project/celebdf_pilot_manifest.json
```

Do this before running the FF++ or Celeb-DF preparation stages.

## Step 4: Prepare FaceForensics++

Return to:

```text
notebooks/00_Project_Setup_and_FFPP_Data_Preparation.ipynb
```

Set:

```python
INSTALL_DEPENDENCIES = True
PREPARE_FFPP_DATA = True
```

Run all cells.

The notebook will:

* download the FaceForensics++ source data through KaggleHub;

* load the supplied video-level split manifest;

* extract ten uniformly distributed frames per source video;

* populate the training, validation, and test directories;

* verify the expected image counts;

* check for source-video overlap between splits.

Do not continue if the final verification reports missing files, unexpected counts, or source-video leakage.

## Step 5: Prepare Face and SBI Inputs

Run the required preprocessing notebooks:

```text
02a_Face_Preprocessing.ipynb
02b_SBI_Preprocessing.ipynb
02c_B4_Test_Face_Preparation.ipynb
```

Their roles are:

* `02a` creates the face-cropped FF++ dataset.

* `02b` prepares the SBI-related resources.

* `02c` prepares the higher-resolution face inputs required by the EfficientNet-B4 experiments.

Only run the preprocessing notebooks required by the experiment being reproduced.

## Step 6: Prepare the Celeb-DF v2 Pilot

Open:

```text
notebooks/05_CelebDF_Pilot_Preparation.ipynb
```

Use:

```python
INSTALL_DEPENDENCIES = True
PREPARE_CELEBDF_PILOT = True
AUDIT_CELEBDF_PILOT = True
FORCE_REBUILD = False
```

Run all cells.

The notebook will:

* load the supplied Celeb-DF pilot manifest;

* download Celeb-DF v2 through KaggleHub when the source videos are required;

* extract ten uniformly distributed frames from each selected video;

* create RetinaFace crops;

* verify the fixed pilot composition.

The final pilot should contain:

```text
20 authentic videos
20 manipulated videos
10 sampled frames per video
400 frames in total
```

Do not enable `FORCE_REBUILD` for normal reproduction.

## Step 7: Reproduce Results Using the Checkpoints

Download the supplied checkpoints using the link in:

```text
checkpoints/README.md
```

Copy them into:

```text
/content/drive/MyDrive/deepfake_project/saved_models/
```

Run the relevant `04*` notebook for internal FaceForensics++ evaluation or the relevant `05*` notebook for Celeb-DF v2 cross-dataset evaluation.

### Internal FF++ Evaluation

```text
04a_E1_FFPP_Evaluation.ipynb
04b_E11_FFPP_Evaluation.ipynb
04c_E2_FFPP_Evaluation.ipynb
04d_E21_FFPP_Evaluation.ipynb
04e_E22_FFPP_Evaluation.ipynb
04f_E3a_FFPP_Evaluation.ipynb
04g_E3b_FFPP_Evaluation.ipynb
04h_E3c_FFPP_Evaluation.ipynb
04i_E4_FSBI_DWT_FFPP_Evaluation.ipynb
04j_E3d_B4_FFPP_Evaluation.ipynb
04k_E5_FreqBlender_B4_FFPP_Evaluation.ipynb
```

### External Celeb-DF v2 Evaluation

```text
05a_E1_CelebDF_Evaluation.ipynb
05b_E11_CelebDF_Evaluation.ipynb
05c_E2_CelebDF_Evaluation.ipynb
05d_E21_CelebDF_Evaluation.ipynb
05e_E22_CelebDF_Evaluation.ipynb
05f_E3a_CelebDF_Evaluation.ipynb
05g_E3b_CelebDF_Evaluation.ipynb
05h_E3c_CelebDF_Evaluation.ipynb
05i_E4_FSBI_DWT_CelebDF_Evaluation.ipynb
05j_E3d_B4_CelebDF_Evaluation.ipynb
05k_E5_FreqBlender_B4_CelebDF_Evaluation.ipynb
```

Each evaluation notebook loads one checkpoint and saves its outputs independently.

## Full Training Workflow

To reproduce model training rather than loading the supplied checkpoints, run the appropriate preprocessing stage followed by the corresponding training notebook:

```text
03a_E1_Baseline_CNN_Training.ipynb
03b_E11_Face_CNN_Training.ipynb
03c_E2_Frozen_EfficientNet_Training.ipynb
03d_E21_Finetuned_EfficientNet_Training.ipynb
03e_E22_Face_EfficientNet_Training.ipynb
03f_E3a_Adapted_SBI_Training.ipynb
03g_E3b_Official_SBI_Training.ipynb
03h_E3c_SBI_SAM_Training.ipynb
03i_E3d_Official_SBI_EfficientNetB4_Training.ipynb
03j_E4_FSBI_DWT_Training.ipynb
03k_E5_Official_FreqBlender_SBI_EfficientNetB4_Training.ipynb
```

Training every configuration is not required to evaluate the supplied checkpoints.

Because training includes stochastic operations, a new training run may not reproduce every reported value exactly even when the same configuration and random seed are used. The supplied checkpoints, manifests, reference results, and plots preserve the principal runs reported in the dissertation.

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

