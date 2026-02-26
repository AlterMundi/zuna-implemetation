"""
ZUNA Processor — Denoise and upsample EEG recordings with ZUNA

Runs the ZUNA foundation model pipeline on a .fif recording:
  1. Preprocess (resample to 256 Hz, filter, epoch)
  2. Inference (denoise + reconstruct/upsample)
  3. Reconstruct to .fif

This script is designed to run on the GPU machine.
Transfer .fif files here, process, and transfer results back.

Usage:
    python zuna_processor.py --input session_001.fif --output enhanced/
    python zuna_processor.py --input session_001.fif --output enhanced/ --bad-channels TP10
    python zuna_processor.py --input session_001.fif --output enhanced/ --upsample-to 10-20 --gpu
"""

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Standard 10-20 montage target channels
# (channels ZUNA will generate in addition to the original 4)
# ─────────────────────────────────────────────

UPSAMPLE_PRESETS = {
    "10-20": [
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
        "C3", "C4", "Cz", "T3", "T4",
        "P3", "P4", "Pz",
    ],
    "frontal": [
        "Fp1", "Fp2", "F3", "F4", "Fz",
    ],
    "none": [],
}

# ─────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Process EEG recording through ZUNA foundation model"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input .fif file path"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for enhanced files"
    )
    parser.add_argument(
        "--bad-channels", nargs="*", default=[],
        help="Channels to mark as bad for reconstruction (e.g., TP10)"
    )
    parser.add_argument(
        "--upsample-to", choices=list(UPSAMPLE_PRESETS.keys()),
        default="none",
        help="Target channel set for upsampling (default: none, denoise only)"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="Use GPU for inference (default: CPU)"
    )
    parser.add_argument(
        "--gpu-device", type=int, default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=50,
        help="Number of diffusion sampling steps (default: 50)"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Generate comparison plots"
    )
    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if not str(input_path).endswith(".fif"):
        print(f"ERROR: Input must be a .fif file, got: {input_path}")
        sys.exit(1)

    # Try to import zuna
    try:
        from zuna import preprocessing, inference, pt_to_fif, compare_plot_pipeline
    except ImportError:
        print("ERROR: 'zuna' package not installed.")
        print("Install it with: pip install zuna")
        print("This script is meant to run on the GPU machine.")
        sys.exit(1)

    # Setup directories
    config = load_config()
    zuna_config = config.get("zuna", {})
    output_dir = Path(args.output)
    working_dir = output_dir / "working"

    fif_filter_dir = str(working_dir / "1_fif_filter")
    pt_input_dir = str(working_dir / "2_pt_input")
    pt_output_dir = str(working_dir / "3_pt_output")
    fif_output_dir = str(output_dir)
    figures_dir = str(output_dir / "figures")

    # Ensure input directory contains just our file
    # ZUNA processes directories, so we create a temp input dir
    input_staging = working_dir / "0_fif_input"
    input_staging.mkdir(parents=True, exist_ok=True)

    import shutil
    staged_input = input_staging / input_path.name
    shutil.copy2(str(input_path), str(staged_input))

    # Determine target channels for upsampling
    target_channels = UPSAMPLE_PRESETS.get(args.upsample_to, [])

    # Device selection
    gpu_device = args.gpu_device if args.gpu else ""

    print(f"\n{'='*50}")
    print(f"  🤖 ZUNA Processor")
    print(f"{'='*50}")
    print(f"  Input:        {input_path}")
    print(f"  Output:       {output_dir}")
    print(f"  Bad channels: {args.bad_channels or 'none'}")
    print(f"  Upsample:     {args.upsample_to} ({len(target_channels)} new channels)")
    print(f"  Device:       {'GPU ' + str(args.gpu_device) if args.gpu else 'CPU'}")
    print(f"  Diff. steps:  {args.diffusion_steps}")
    print(f"{'='*50}\n")

    # ─── Step 1: Preprocess ───
    print("Step 1/3: Preprocessing...")
    preprocessing(
        input_dir=str(input_staging),
        output_dir=pt_input_dir,
        apply_notch_filter=zuna_config.get("apply_notch_filter", False),
        apply_highpass_filter=zuna_config.get("apply_highpass_filter", True),
        apply_average_reference=zuna_config.get("apply_average_reference", True),
        target_channel_count=target_channels if target_channels else None,
        bad_channels=args.bad_channels if args.bad_channels else None,
        preprocessed_fif_dir=fif_filter_dir,
    )
    print("  ✓ Preprocessing complete\n")

    # ─── Step 2: Inference ───
    print("Step 2/3: Running ZUNA inference...")
    inference(
        input_dir=pt_input_dir,
        output_dir=pt_output_dir,
        gpu_device=gpu_device,
        data_norm=zuna_config.get("data_norm", 10.0),
        diffusion_sample_steps=args.diffusion_steps,
        diffusion_cfg=1.0,
        plot_eeg_signal_samples=False,
        inference_figures_dir=figures_dir,
    )
    print("  ✓ Inference complete\n")

    # ─── Step 3: Reconstruct .fif ───
    print("Step 3/3: Reconstructing .fif...")
    pt_to_fif(
        input_dir=pt_output_dir,
        output_dir=fif_output_dir,
    )
    print("  ✓ Reconstruction complete\n")

    # ─── Optional: Comparison plots ───
    if args.plot:
        print("Generating comparison plots...")
        compare_plot_pipeline(
            input_dir=str(input_staging),
            fif_input_dir=fif_filter_dir,
            fif_output_dir=fif_output_dir,
            pt_input_dir=pt_input_dir,
            pt_output_dir=pt_output_dir,
            output_dir=figures_dir,
            plot_pt=True,
            plot_fif=True,
            num_samples=2,
        )
        print(f"  ✓ Plots saved to {figures_dir}\n")

    # List output files
    output_fifs = list(Path(fif_output_dir).glob("*.fif"))
    print(f"{'='*50}")
    print(f"  Processing complete!")
    print(f"  Output files:")
    for f in output_fifs:
        print(f"    {f}")
    print(f"{'='*50}")
    print(f"\n  Transfer back with:")
    print(f"  scp {output_fifs[0] if output_fifs else output_dir/'*.fif'} local-machine:~/enhanced/")


if __name__ == "__main__":
    main()
