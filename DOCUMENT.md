# DOCUMENT.md

<!--
This file lives in the root of every forked repo.
Fill it in as you go. Do not reconstruct it after the fact.
Keep entries factual and brief. The audience is a future person
reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

<!--
Copy this information from the upstream repo's README and paper.
"Paired or unpaired" refers to whether the model assumes paired training data.
If the model is domain-specific to virtual staining, note the exact staining task (e.g. H&E to HER2 IHC).
-->

- **Model name:** ASP (Adaptive Supervised PatchNCE)
- **Upstream repo URL:** https://github.com/lifangda01/AdaptiveSupervisedPatchNCE
- **Fork URL:** https://github.com/InViLabVirtualStainingBenchmark/AdaptiveSupervisedPatchNCE
- **Upstream last commit date:** Jan 7, 2025
- **Paper / citation:** https://arxiv.org/abs/2303.06193 (MICCAI 2023)
- **Paired or unpaired assumption:** Paired
- **Intended staining task (if domain-specific):** H&E-to-IHC (HER2, Ki67, ER, PR)

---

## Environment Claimed by Authors

<!--
Record exactly what the authors say in their README or requirements file.
Do not adjust or interpret -- copy their stated versions.
"Requirements file present" should note the filename if it exists.
If no version is specified for Python or PyTorch, write "not specified".
-->

- **Python version:** 3.9
- **PyTorch version:** 1.12.1
- **CUDA version:** 11.6
- **Installation method:** conda
- **Requirements file present:** environment.yml
- **Pretrained weights available:** yes
- **Pretrained weights notes:**
    - Hosted on Google Drive. GDrive links can rot.
<!-- Where are they hosted? Are they behind a login? Is the link likely to rot (GDrive, Dropbox, personal server)? -->

---

## Environment Actually Used

<!--
Record the environment you actually created and tested in.
If you deviated from what the authors specified, briefly note why (e.g. "authors' version not compatible with CUDA 12.1").
Conda env name should follow the convention: the model's short name.
-->

- **Python version:** 3.9.25
- **PyTorch version:** 2.5.1
- **CUDA version:** 12.1
- **Conda environment name:** asp
- **Date tested:** 06.05.2026
- **Hardware:** RTX 4090, WSL2 on Windows 11

---

## Installation

<!--
Follow the authors' README exactly before making any changes.
Record the commands you ran in order.
If an error occurred, paste the key line of the error (not the full traceback) and then record the fix.
If installation succeeded without issues, write "No issues."
-->

### Commands Run

```bash
# paste the installation commands here in order
conda create -n asp python=3.9 -y
conda activate asp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install scipy dominate opencv-python Pillow numpy visdom packaging GPUtil
```

- **Deviations from authors' instructions:**
    - Did not use environment.yml
    - PyTorch 2.5.1 + CUDA 12.1 used instead of PyTorch 1.12.1 + CUDA 11.6 to match RTX 4090 hardware

### Issues and Fixes

<!--
Format: problem encountered -> fix applied.
If no issues, write "None."
-->

| Issue | Fix Applied |
| --- | --- |
|  |  |

### GPU Confirmation

<!--
Paste the output of the check below so there is proof the GPU was visible.
Command: python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
-->

```
# paste output here
```

---

## Dataset Preparation

<!--
Record how the dataset was prepared for this specific model.
"Format expected" means what folder layout or file structure the model's data loader assumes
(e.g. side-by-side paired images, separate A/B folders, CSV manifest, etc.).
"Conversion applied" means any script or command you ran to reformat the standard BCI/MIST-HER2
download into the format this model needs.
If no conversion was needed, write "None -- dataset used as downloaded."
-->

- **Dataset used:** BCI & MIST-HER2
- **Format expected by model:** 
- **Conversion applied:**
    
    ```bash
    # paste conversion command(s) here if any
    ```
    
- **Final folder layout used:**
    
    ```
    # BCI:
    #     trainA/trainB: 100 images (first 100 of 3896)
    #     testA/testB:   20 images (first 20 of 977)
    # 
    # MIST:
    #     trainA/trainB: 100 images (first 100 of 4642)
    #     valA/valB:     20 images (first 20 of 1000)
    # 
    # Note: MIST uses valA/valB instead of testA/testB.
    # Data loader falls back to valA/valB automatically when phase=test and testA is absent.
    ```
    
- **Number of images used for smoke test (train / test):**

---

## Pretrained Weights

<!--
Only fill this section if pretrained weights exist.
Record the exact download source. Flag any link that is not on a stable host
(Zenodo and HuggingFace are stable; Google Drive, Dropbox, and personal servers are at risk).
Record where you placed the weights relative to the repo root.
-->

- **Download source URL:** [Google Drive](https://drive.google.com/drive/folders/11a3_4cyQY1bgBiRKnqtM7JGis5CPoVM6)
- **Host stability:** at-risk (GDrive)
- **Weights placed at (relative path):**
- **Size on disk:**

---

## Inference Smoke Test

<!--
Run inference before training if pretrained weights are available -- it is faster
and confirms the code path works independently of the training loop.
Use 10-20 images from the BCI or MIST-HER2 test split.
"Visual check" is a qualitative sanity check only -- not a metric.
Valid outcomes: "images look like expected domain", "blank/grey output", "wrong resolution", "file not written".
-->

- **Script / command run:**
    
    ```bash
    # BCI
    # Inference
    python test.py --dataroot ~/internship-models/datasets/asp/BCI-asp --name bci_her2_lambda_linear --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --normG instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 1024 --preprocess none --phase test --num_test 20
    # Evaluate
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/AdaptiveSupervisedPatchNCE/results/bci_her2_lambda_linear/test_latest/images/fake_B --gt ~/internship-models/datasets/asp/BCI-asp/testB --model_name asp --dataset_name BCI --split_name test --match_by stem --output ~/internship-models/results.csv
    # MIST-HER2

    # Inference
    python test.py --dataroot ~/internship-models/datasets/asp/MIST-asp --name mist_her2_lambda_linear --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --normG instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 1024 --preprocess none --phase test --num_test 20
    # Evaluate
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/AdaptiveSupervisedPatchNCE/results/mist_her2_lambda_linear/test_latest/images/fake_B --gt ~/internship-models/datasets/asp/MIST-asp/valB --model_name asp --dataset_name MIST-HER2 --split_name val --match_by stem --output ~/internship-models/results.csv
    ```
    
- **Output folder:**
    - BCI: ./results/bci_her2_lambda_linear/test_latest/images/fake_B/
    - MIST-HER2: ./results/mist_her2_lambda_linear/test_latest/images/fake_B/
- **Number of output images produced:** 20 (BCI), 20 (MIST-HER2)
- **Output image dimensions:** 1024x1024 (preprocess=none, full image passed through)
- **Visual check result:** outputs show IHC-like color distribution (light background,
DAB-like brown staining, tissue structures visible). PASS.
- **Time to run (approx):** < 1 min for 20 images on RTX 4090
- **Errors or warnings during inference:**
<!-- "None" if clean. Otherwise paste the key error line. -->

---

## Training Smoke Test

<!--
Run training for 5 epochs minimum. The goal is a clean exit, not a useful model.
Use the smallest viable batch size and the model's default resolution unless that causes an OOM error.
Always set checkpoint saving to every epoch (e.g. --save_epoch_freq 1 for pix2pix-style repos)
so there is proof a checkpoint was written.
Monitor GPU memory with: watch -n 1 nvidia-smi (run in a second terminal).
-->

- **Script / command run:**
    
    ```bash
    # BCI
    # Train
    python train.py --dataroot ~/internship-models/datasets/asp/BCI-asp --name asp_bci_smoketest --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --netD n_layers --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 512 --preprocess crop --batch_size 1 --n_epochs 3 --n_epochs_decay 2 --save_epoch_freq 1 --display_id -1 --num_threads 4
    # Inference
    python test.py --dataroot ~/internship-models/datasets/asp/BCI-asp --name asp_bci_smoketest --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --normG instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 1024 --preprocess none --phase test --num_test 20
    # Evaluate
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/AdaptiveSupervisedPatchNCE/results/asp_bci_smoketest/test_latest/images/fake_B --gt ~/internship-models/datasets/asp/BCI-asp/testB --model_name asp --dataset_name BCI --split_name test --match_by stem --output ~/internship-models/results.csv

    # MIST-HER2
    # Train
    python train.py --dataroot ~/internship-models/datasets/asp/MIST-asp --name asp_mist_smoketest --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --netD n_layers --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 512 --preprocess crop --batch_size 1 --n_epochs 3 --n_epochs_decay 2 --save_epoch_freq 1 --display_id -1 --num_threads 4
    # Inference
    python test.py --dataroot ~/internship-models/datasets/asp/MIST-asp --name asp_mist_smoketest --checkpoints_dir ./checkpoints --model cpt --CUT_mode FastCUT --dataset_mode aligned --direction AtoB --netG resnet_6blocks --normG instance --weight_norm spectral --no_dropout --nce_layers 0,4,8,12,16 --load_size 1024 --crop_size 1024 --preprocess none --phase test --num_test 20
    # Evaluate
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/AdaptiveSupervisedPatchNCE/results/asp_mist_smoketest/test_latest/images/fake_B --gt ~/internship-models/datasets/asp/MIST-asp/valB --model_name asp --dataset_name MIST-HER2 --split_name val --match_by stem --output ~/internship-models/results.csv
    ```
    
- **Dataset used:**
    - BCI: BCI-asp (100 train images, 20 test images, symlinked from original)
    - MIST-HER2: MIST-asp (100 train images, 20 val images, symlinked from original)
- **Epochs run:** 5 (n_epochs=3 + n_epochs_decay=2)
- **Batch size:** 1
- **Input resolution:** 512x512 (cropped from 1024x1024 at load time)
- **Time per epoch (approx):** 15-24 sec on RTX 4090
- **Peak GPU memory (approx, from nvidia-smi):** not recorded
- **Checkpoint saved:** yes
- **Checkpoint path:**
    - BCI: ./checkpoints/asp_bci_smoketest/
    - MIST-HER2: ./checkpoints/asp_mist_smoketest/
    - Files: 1_net_G.pth through 5_net_G.pth, latest_net_G.pth (same for D and F)
- **Crash or error during training:**
<!-- "None" if clean. Otherwise paste the key error line and the fix applied. -->

---

## Output Verification

<!--
Open 3-5 output images and compare them visually against the ground-truth target.
This is not a metric -- just a check that the model produced something in the right domain.
"Expected domain" for BCI would be IHC HER2-stained tissue with brown DAB staining on a light background.
Record one or two example output filenames so the check is reproducible.
-->

- **Output folder:**
    - BCI: ./results/asp_bci_smoketest/test_latest/images/fake_B/
    - MIST-HER2: ./results/asp_mist_smoketest/test_latest/images/fake_B/
- **Example output filenames:**
    - BCI: 00000_test_1+.png, 00001_test_2+.png, 00002_test_2+.png
    - MIST-HER2: 100M2004069_10_9.jpg, 100M2004069_19_30.jpg
- **Dimensions match input:** yes
- **Visual sanity check:** outputs show IHC-like staining, structures roughly aligned with H&E input. Color distribution consistent with HER2 IHC target domain.
<!-- e.g. "outputs show IHC-like staining, structures roughly aligned with H&E input" -->
- **Any obvious artifacts or failure modes:** None noted at 5 epochs. Quality is expected to be low (this is a smoke test only.)

---

## Changes Made to Original Code

<!--
Record every change made to the original repo, no matter how small.
Do not make changes that alter model architecture or training logic.
Only changes needed for the code to run in the benchmark environment are allowed.
Add rows as needed.
-->

| File | Change Description | Reason |
| --- | --- | --- |
|  |  |  |
|  |  |  |

<!--
Common examples of acceptable changes:

- Pinning a dependency version in requirements.txt (e.g. torch==2.1.0) because no version was specified
- Replacing a hardcoded absolute path with a command-line argument
- Removing an import that is not used and is not installable in the benchmark environment
- Adapting the data loader to accept BCI/MIST-HER2 folder structure
-->

---

## Frozen Environment

<!--
After the smoke test passes, export and commit the environment file.
Command: conda env export > environment_<model-name>.yml
This file is what gets adapted for the HPC migration later.
Note any packages that are unusual, very large, or likely to cause conflicts on the cluster.
-->

- **Environment file:** `environment_asp.yml`
- **Committed to fork:** yes
- **Notes on unusual or heavy dependencies:**
<!-- e.g. "requires openslide-python which needs a system-level apt install" -->

---

## HPC Readiness Notes

<!--
Fill this in after the local smoke test passes.
Flag anything that will need attention before running on the VSC cluster.
Common issues: GUI/display dependencies (matplotlib backends), hardcoded CUDA package versions,
dependencies that require apt/system installs, very large model downloads.
Leave blank until local test is complete.
-->

- **Display/GUI dependencies to remove or neutralize:**
- **System-level dependencies (non-pip/conda):**
- **Estimated GPU memory requirement:**
- **Estimated storage requirement (weights + data):**
- **Other notes for cluster adaptation:**

---

## Summary

<!--
Write 2-4 sentences summarizing what worked, what did not, and what the next step is.
Be specific. Include the overall pass/fail verdict.
This is the first thing someone reads when picking this model back up.
-->

**Overall result:** PASS

<!-- Example pass:
"[Model] smoke test completed on [date]. Inference with pretrained weights passed on 10 BCI test images.
Training ran for 5 epochs without crash. One change was made to the data loader to accept separate
source/target folders. Frozen environment committed. Ready for full benchmark run."

Example fail:
"[Model] smoke test failed at the environment step. The required PyTorch version (1.4) is not
compatible with CUDA 12.1. Blocked until a workaround is identified. Do not schedule for HPC."
-->