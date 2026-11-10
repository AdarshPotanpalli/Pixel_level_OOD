from src.train import train
import torch as t
import argparse

parser = argparse.ArgumentParser(description= "CLI configurations for JeT training")
parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs"
)
parser.add_argument(
    "--block_depth",
    type=int,
    default=2,
    help="Depth of inner block subnets",
)
parser.add_argument(
    "--coupling_layers",
    type=int,
    default=8,
    help="Number of coupling layers",
)
parser.add_argument(
    "--tag_name",
    type=str,
    default="",
    help="Experiment tag name appended to checkpoints & metric logs",
)

args = parser.parse_args()

device = "cuda" if t.cuda.is_available() else "cpu"

jet_config = dict(
    patch_size=1,
    patch_dim=384,
    n_patches=9,
    coupling_layers=args.coupling_layers,
    block_depth=args.block_depth,
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
        epochs=args.epochs,
        warmup_percentage=0.1,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        weight_decay=1e-5,
        adam_betas=(0.9, 0.95),
        features_path_train="dataset/cityscapes/cityscapes_dinov3_vits_features/train",
        features_path_valid="dataset/dataset_AnomalyTrack/AnomalyTrack_dinov3_vits_features/images",
        num_workers=2, # during debug set it to 1, else 2
        patch_stride=3, # stride for getting 3X3 patch grid using unfold
        checkpoint_path= f"jet_dino_{args.tag_name}.pt", # checkpoint path if all epochs are completed
        best_checkpoint_path=f"best_jet_dino_{args.tag_name}.pt", # incase of early stopping, the best model out of last 5 epochs will be saved here
        log_dir="logs",
        patience=5, # early stopping patience
        min_delta=1e-4, # min decrease in val_bpd to not increase the patience counter
        metrics= f"metrics_{args.tag_name}"
    )