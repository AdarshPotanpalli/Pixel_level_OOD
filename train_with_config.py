from src.train import train
import torch as t

device = "cuda" if t.cuda.is_available() else "cpu"

jet_config = dict(
    patch_size=1,
    patch_dim=384,
    n_patches=9,
    coupling_layers=16,
    block_depth=4,
    block_width=192, # default block width selected
    num_heads=3, # default num heads selected
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

if __name__ == "__main__":
    train(
        jet_config=jet_config,
        image_batch_size=32, # so that we run unfold on a image_batch_size images at once
        batch_size=512,
        accumulate_steps=16,
        device=device,
        epochs=100,
        warmup_percentage=0.1,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        weight_decay=1e-5,
        adam_betas=(0.9, 0.95),
        features_path_train="dataset/cityscapes/cityscapes_dinov3_vits_features/train",
        features_path_valid="dataset/cityscapes/cityscapes_dinov3_vits_features/val",
        num_workers=1, # during debug set it to 1, else 2
        patch_stride=3, # stride for getting 3X3 patch grid using unfold
        checkpoint_path="jet_dino.pt", # checkpoint path if all epochs are completed
        best_checkpoint_path="best_jet_dino.pt", # incase of early stopping, the best model out of last 5 epochs will be saved here
        log_dir="logs",
        patience=5, # early stopping patience
        min_delta=1e-4, # min decrease in val_bpd to not increase the patience counter
    )