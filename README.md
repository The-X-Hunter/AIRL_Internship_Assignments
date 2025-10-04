# Internship Coding Assignment

This repository contains solutions for **Q1 (Vision Transformer on CIFAR-10)** and **Q2 (Text-driven Image/Video Segmentation using GroundingDINO + SAM2/SAM2Video)**.  

---

## 📘 Q1: Vision Transformer (ViT) on CIFAR-10

### 🚀 How to Run in Colab
1. Upload `q1.ipynb` to [Google Colab](https://colab.research.google.com/).  
2. In Colab:  
   - Go to `Runtime > Change runtime type`  
   - Select **T4 GPU**  
   - Click **Run All**  

---

### ⚙️ Best Model Configuration
- **Patch Size** = `2 × 2` (finer detail capture)  
- **Epochs** = `150` (ViTs are data-hungry, require more training)  
- **Depth** = `6` transformer blocks  
- **Batch Size** = `128`  
- **Learning Rate** = `3e-4`  
- **Embedding Dimension** = `256`  
- **No. of Attention Heads** = `8` (divides embedding dimension evenly)  
- **MLP Dimension** = `512`  

---

### 🌱 Why & How Augmentation?
- **Why**: ViTs lack convolutional inductive bias → prone to overfitting. Data augmentation improves generalization.  
- **How**:  
  - `RandomCrop(32, padding=4)` → adds translation invariance.  
  - `RandomHorizontalFlip()` → handles mirror symmetry.  
  - `RandAugment` → stronger variations in brightness, contrast, color.  

---

### 📉 Why & How Scheduler?
- **Why**: Fixed learning rate stalls training. Cosine annealing + warmup helps stabilize training and prevents early convergence.  
- **How**:  
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
  for epoch in range(EPOCHS):
      train(...)
      validate(...)
      scheduler.step()

---

### Label Smoothening
- When training a model, especially in classification tasks, the model often learns to predict highly confident outputs, i.e., it assigns almost all of the probability mass to a single class.

---

### Result
- Due to Google Colab session expiring issue, I had to run it on other system. Other than that there is no issue executing the notebook on Colab.
- For accuracies noted on each epochs, kindly refer to noted_acc.json

| Model Config | Patch Size | Epochs | Depth | Batch Size | LR   | Embedding Dim | Heads | MLP Dim | Test Accuracy |
|--------------|------------|--------|-------|------------|------|---------------|-------|---------|---------------|
| Best Model   | 2×2        | 150    | 6     | 128        | 3e-4 | 256           | 8     | 512     | ~80% |

---

### Other possible tweak
- Using timm pretrained ViTs on ImageNet then finetuning on CIFAR-10 can get better results.

---

# Q2: Text-driven Image & Video Segmentation

This folder contains the implementation of **text-driven image and video segmentation** using **GroundingDINO + SAM2/SAM2Video**.

---

### 🚀 How to Run in Colab
1. Upload `q2.ipynb` to [Google Colab](https://colab.research.google.com/).  
2. In Colab:  
   - Go to `Runtime > Change runtime type`  
   - Select **T4 GPU**  
   - Click **Run All**  

---

## 🏗️ Pipeline Overview

1. **User Input**
   - Accepts either an **image** or a **video**.
   - *(Can be extended to dynamically ask for input path & text prompt from the user)*

2. **GroundingDINO**
   - Input: Image + text prompt (e.g., `"a dog"`)
   - Output: Region seeds → bounding box coordinates for the detected object.

3. **SAM2 (Image Segmentation)**
   - Input: Image + bounding box
   - Output: Segmentation mask
   - Mask is overlaid on the original image for visualization (and can be saved).

4. **SAM2Video (Video Segmentation)**
   - Load video frames using **OpenCV**.
   - Pass **first frame + text prompt** → GroundingDINO → bounding box.
   - Video frames + bounding box → SAM2Video → segmentation initialized at frame 0.
   - SAM2Video propagates segmentation across the entire video.
   - Output: New video with segmentation overlays.

---

## ⚠️ Limitations

- **First-frame dependency**: If GroundingDINO misdetects the object in the first frame, the propagated segmentation will be poor.
- **Prompt sensitivity**: Results vary depending on clarity of the text prompt.
- **Resource intensive**: Large models → slow on CPU. GPU (T4/A100) recommended.
- **Single-object support**: Current pipeline only handles one object per run.
