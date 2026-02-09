# Industrial Vision Robustness: A Human-Centric AI Approach for Fault Diagnosis

This repository contains the preliminary exploration for a Master's research project aimed at building robust automated inspection systems for industrial components. The project bridges signal processing, computer vision, and human-centric AI to address the challenges of "Harsh Industrial Environments."

## 📌 Research Vision: The Two-Phase Framework
This study is structured into two strategic phases to align industrial maintenance needs with advanced computer vision research:

* **Phase 1: Human-in-the-Loop & Gaze-Guided Attention**
    Investigating how human expert intuition can be modeled through gaze-tracking data to improve the interpretability and focus of defect detection systems.
* **Phase 2: Contrastive Learning for Environmental Robustness**
    Developing a self-supervised framework (e.g., SimCLR/MoCo) to ensure AI reliability when faced with data scarcity and sensory noise on real-world factory floors.

---

## 🛠️ Technical Workflow
The following pipeline describes the transformation of industrial sensor data into robust visual embeddings.

| Stage | Input/Data | Method | Purpose |
| :--- | :--- | :--- | :--- |
| **Data Preparation** | 99 `.mat` files (HUST Dataset) | `scipy.io` Parsing | Extracting high-frequency vibration signals and dynamic sampling rates ($fs$). |
| **Imaging** | 1D Raw Signals | **STFT (Short-Time Fourier Transform)** | Generating 2D Spectrograms as the primary visual input for the CV model. |
| **Degradation** | Clean Spectrograms | **Albumentations Library** | Simulating "Harsh Environments" via Gaussian noise, motion blur, and data dropout. |
| **Feature Analysis** | Visual Spectrograms | **Pre-trained ResNet18** | Extracting deep spatial-temporal features (embeddings) from the images. |
| **Comparison** | Feature Vectors | **Cosine Similarity** | Benchmarking model stability by comparing Clean vs. Harsh feature signatures. |



---

## 📊 Preliminary Results Analysis
A benchmark test was conducted using the **OB702.mat** sample (Outer Race Fault). This experiment evaluates how standard deep learning features react to simulated industrial interference.

### Experimental Sample: `OB702.mat`
* **Original State**: Clear frequency harmonics representing a structural bearing fault.
* **Simulated Harsh State**: Introduced heavy sensor noise (Gaussian) and camera vibration (Motion Blur) to mimic factory conditions at companies like Yokogawa or Mitsubishi Power.

**Resulting Metric:**
* **Cosine Similarity Score: 0.8433**

This score indicates that while the model retains **84.33%** of the core fault identity, there is a **~16% perception gap** caused by environmental noise. This gap represents the primary research problem to be addressed during the Master's program.



---

## 🚀 Future Roadmap & Master’s Research Goals
The results identified in this exploration serve as the foundation for the upcoming Master's thesis at Tottori University. The research will evolve through:

1.  **Primary Data Acquisition**: 
    Collecting real physical samples of defective bearings and capturing high-resolution macro imagery to simulate real-world QC (Quality Control) lines.
    
2.  **Implementation of Gaze-Guided Attention**: 
    Utilizing the laboratory’s **Eye-Tracking infrastructure** to capture expert visual search patterns, integrating them as a "prior" to enhance anomaly localization.
    
3.  **Self-Supervised Contrastive Learning**: 
    Training the architecture to learn noise-invariant representations. By optimizing the model to recognize that a "noisy" image and a "clean" image of the same defect are identical in the feature space, we aim to push the Similarity Score closer to **1.0**.



---

## ⚙️ How to Reproduce
1.  Clone this repository.
2.  Place the HUST dataset in the data folder.
3.  Install dependencies: 
    `pip install scipy numpy matplotlib opencv-python albumentations torch torchvision scikit-learn`
4.  Run the exploration script:
    ```bash
    python exploration.py
    ```

---
