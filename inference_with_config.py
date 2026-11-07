from src.inference import run_patch_inference
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

INFERENCE_OVER_A_SINGLE_IMAGE = False

jet_config = dict(
    patch_size=1,
    patch_dim=384,
    n_patches=9,
    coupling_layers=16,
    block_depth=4,
    block_width=192,
    num_heads=3,
    scale_factor=2.0,
    coupling_types=(
        "channels", "channels",
        "channels", "channels",
        # "spatial",
    ),
    spatial_coupling_projs=(
        "checkerboard", "checkerboard-inv",
        "vstripes", "vstripes-inv",
        "hstripes", "hstripes-inv",
    )
)

def inference_and_interpolation(JET_CHECKPOINT:str, SAMPLE_IMAGE:str, output_png:str):
    # 1. Unpack metrics, losses, and the structural patch geometry
        bpds, nlls, logits, logdets, out_shape = run_patch_inference(
            image_path=SAMPLE_IMAGE,
            jet_model_path=JET_CHECKPOINT,
            jet_config=jet_config,        # FIX: Pass your config here!
            patch_batch_size=512,
            hf_token= os.getenv("HF_TOKEN") 
        )
        
        # 2. Reshape into the valid sliding-window grid (62x126)
        spatial_bpd_map = bpds.reshape(out_shape)
        
        # 3. Add Padding to restore the clipped boundaries (62x126 -> 64x128)
        # pad=(left, right, top, bottom) -> adds 1 cell everywhere using edge replication
        # We add unsqueeze blocks because F.pad expects batch/channel dimensions (B, C, H, W)
        padded_features = F.pad(
            spatial_bpd_map.unsqueeze(0).unsqueeze(0), 
            pad=(1, 1, 1, 1), 
            mode="replicate"
        )
        
        # 4. Interpolate 16x up to the original native size (1024, 2048)
        # Using align_corners=False prevents structural shifting artifacts at pixel edges
        native_heatmap_tensor = F.interpolate(
            padded_features, 
            scale_factor=16, 
            mode="bilinear", 
            align_corners=False
        ).squeeze() # Shape is now exactly (1024, 2048)
        
        # Convert tensor to a standard numpy matrix for plotting
        heatmap_numpy = native_heatmap_tensor.cpu().numpy()
    
        print(f"\n[SUMMARY] Image Processing Complete:")
        print(f"  - Valid Unfold Shape:     {out_shape}")
        print(f"  - Restored DINO Grid:     {list(padded_features.shape[-2:])}")
        print(f"  - Final Heatmap Matrix:   {heatmap_numpy.shape}")
    
        # ==========================================
        # 5. Draw and Save the Visualization Heatmap
        # ==========================================
        orig_img = Image.open(SAMPLE_IMAGE).convert("RGB")
        
        plt.figure(figsize=(15, 7))
        
        # Background: Plot original Cityscapes image
        plt.imshow(orig_img)
        
        # Foreground: Overlay BPD map as a semi-transparent layer
        # 'jet' or 'inferno' colormaps work well for BPD variations (blue=normal, red=anomaly)
        plt.imshow(heatmap_numpy, cmap="jet", alpha=0.5) 
        
        plt.colorbar(label="Bits Per Dimension (BPD Loss Score)")
        plt.title("JeT Anomaly Prediction Distribution Heatmap (Cityscapes Native)")
        plt.axis("off")
        
        # Save output visualization graph
        plt.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  - Successfully saved visualization heatmap array to: {output_png}")

if __name__ == "__main__":
    SAMPLE_IMAGE = "dataset/dataset_AnomalyTrack/images/validation0000.jpg"
    JET_CHECKPOINT = "src/jet_dino.pt"
    
    sample_images = ["dataset/dataset_AnomalyTrack/images/validation0000.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0001.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0002.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0003.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0004.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0005.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0006.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0007.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0008.jpg",
                     "dataset/dataset_AnomalyTrack/images/validation0009.jpg"]
    
    if INFERENCE_OVER_A_SINGLE_IMAGE:
        output_filename = f"{Path(SAMPLE_IMAGE).stem}_overlay.png"
        inference_and_interpolation(JET_CHECKPOINT=JET_CHECKPOINT, 
                                    SAMPLE_IMAGE= SAMPLE_IMAGE,
                                    output_png= output_filename)
    else:
        for sample_image in sample_images:
            output_filename = f"{Path(sample_image).stem}_overlay.png"
            inference_and_interpolation(JET_CHECKPOINT=JET_CHECKPOINT, 
                                        SAMPLE_IMAGE= sample_image,
                                        output_png= output_filename)

    



