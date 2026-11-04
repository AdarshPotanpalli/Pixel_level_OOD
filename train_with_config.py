from src.train import train

jet_config = dict(
    patch_size=1,
    patch_dim=384,
    n_patches=9,
    coupling_layers=16,
    block_depth=4,
    block_width=512,
    num_heads=8,
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
        batch_size=64,
        accumulate_steps=16,
        device="cpu",
        epochs=100,
        warmup_percentage=0.1,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        weight_decay=1e-5,
        adam_betas=(0.9, 0.95),
        features_path_train="dataset/cityscapes/cityscapes_dinov3_vits_features/train",
        features_path_valid="dataset/cityscapes/cityscapes_dinov3_vits_features/val",
        num_workers=4, # during debug set it to 1
        checkpoint_path="jet_dino.pt",
    )