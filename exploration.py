import os
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import torch
import torchvision.models as models
import torchvision.transforms as T
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ==========================================
# 1. config PATH
# ==========================================
DATASET_PATH = r"D:\BEASISWA\MEXT\fujimoto tadahiro\hust bearing dataset\dataset"
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2.  DATA PREP & IMAGING 
# ==========================================
def signal_to_spectrogram(file_path):
    """Membaca file .mat dengan key 'data' dan 'fs'"""
    mat = scipy.io.loadmat(file_path)
    
    # vibration data (512000, 1) to 1D
    raw_signal = mat['data'].flatten()
    
    # dynamic sampling frequency from file
    sampling_freq = mat['fs'][0][0]
    
    # Take a signal sample (e.g., 4096 points for better resolution)
    segment = raw_signal[:4096]
    
    # Transformation STFT (Short-Time Fourier Transform)
    f, t, Sxx = signal.spectrogram(segment, fs=sampling_freq)
    
    # convert to log scale (dB)
    spectrogram_db = 10 * np.log10(Sxx + 1e-10)
    
    # normalization range 0-255 (Grayscale)
    norm_spec = cv2.normalize(spectrogram_db, None, 0, 255, cv2.NORM_MINMAX)
    return norm_spec.astype(np.uint8)

# ==========================================
# 3. DEGRADATION (HARSH ENVIRONMENT)
# ==========================================
def apply_industrial_noise(image):
    """Simulasi gangguan lingkungan industri"""
    transform = A.Compose([
        A.GaussNoise(var_limit=(100, 500), p=0.8),      # noise sensor
        A.MotionBlur(blur_limit=7, p=0.6),             # vibration camera
        A.RandomBrightnessContrast(p=0.5),             # not stable light 
        A.CoarseDropout(max_holes=4, max_height=30, max_width=30, p=0.4) # Data loss/interference
    ])
    return transform(image=image)['image']

# ==========================================
# 4. COMPARISON (FEATURE EXTRACTION)
# ==========================================
class FeatureExtractor:
    def __init__(self):
        # ResNet18 for visual feature extraction
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        self.preprocess = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_vector(self, img_array):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_t = self.preprocess(img_pil).unsqueeze(0)
        with torch.no_grad():
            features = self.model(img_t)
        return features.flatten().numpy().reshape(1, -1)

# ==========================================
# execution
# ==========================================
extractor = FeatureExtractor()
mat_files = glob.glob(os.path.join(DATASET_PATH, "*.mat"))

if not mat_files:
    print("Folder kosong atau path salah!")
else:
    # Testing on the file (change [] inside mat_files)
    sample_file = mat_files[94]
    file_name = os.path.basename(sample_file)
    print(f"Memproses file: {file_name}")

    # A. Imaging
    clean_img = signal_to_spectrogram(sample_file)

    # B. degradation
    harsh_img = apply_industrial_noise(clean_img)

    # C. Comparison
    vec_clean = extractor.get_vector(clean_img)
    vec_harsh = extractor.get_vector(harsh_img)
    sim_score = cosine_similarity(vec_clean, vec_harsh)[0][0]

    # visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(clean_img, cmap='magma')
    plt.title(f"Clean (Source: {file_name})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(harsh_img, cmap='magma')
    plt.title(f"Harsh Simulation\nSimilarity: {sim_score:.4f}")
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"result_{file_name.replace('.mat', '.png')}")
    plt.savefig(output_path)
    plt.show()

    print(f"Done! Similarity score: {sim_score:.4f}")
    print(f"Image saved in: {output_path}")