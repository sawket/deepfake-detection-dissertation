# Pre-trained Model Checkpoints

The trained model checkpoints used in this dissertation are stored separately in OneDrive because several files exceed GitHub's standard file-size limit.

Download the checkpoints from:

[Deepfake Dissertation Checkpoints](https://stummuac-my.sharepoint.com/:f:/r/personal/25947826_stu_mmu_ac_uk/Documents/Deepfake%20Dissertation%20Checkpoints?d=w27906e13916e4808b54c95043276068b&csf=1&web=1&e=gKjJYC)

After downloading, copy all checkpoint files into the following Google Drive directory:

```text
/content/drive/MyDrive/deepfake_project/saved_models/
```

The directory should contain the following files:

```text
baseline_cnn.pth
baseline_cnn_face.pth
efficientnet_b0.pth
efficientnet_b0_finetuned.pth
efficientnet_b0_face_finetuned.pth
efficientnet_b0_sbi_best.pth
efficientnet_b0_official_sbi_best.pth
efficientnet_b0_official_sbi_sam_best.pth
efficientnet_b4_official_sbi_best.pth
efficientnet_b0_fsbi_dwt_best.pth
efficientnet_b4_freqblender_sbi_best.pth
```

The evaluation notebooks expect these exact filenames and locations. Do not rename the checkpoint files.

Before running an evaluation notebook, confirm that its required checkpoint exists in the `saved_models` directory.
