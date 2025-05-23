import os
import argparse
from objectstitch import ObjectStitchModel
from utils.data_loader import load_masked_pairs
from utils.noise import apply_mask_noise
from utils.metrics import compute_lpips, compute_fid

def run_with_noise(Modell, image_pair, noise_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    noisy_outputs = []

    for i, (fg, bg, mask) in enumerate(image_pair):
        noisy_mask = apply_mask_noise(mask, noise_type)
        composite = Modell.compose(fg, bg, noisy_mask)
        out_path = os.path.join(output_dir, f"{noise_type}_{i}.png")
        composite.save(out_path)
        noisy_outputs.append(composite)

    return noisy_outputs

def main(args):
    model = ObjectStitchModel.load_pretrained(args.model_path)
    image_pairs = load_masked_pairs(args.data_dir)  
    all_outputs = {}
    for noise in ['gaussian', 'dilate', 'erode']:
        print(f"Running with {noise} mask noise...")
        outputs = run_with_noise(model, image_pairs, noise, os.path.join(args.output_dir, noise))
        all_outputs[noise] = outputs

    # Evaluation are here we can check below
    print("Evaluating with LPIPS and FID...")
    for noise, outputs in all_outputs.items():
        lpips = compute_lpips(outputs, [bg for _, bg, _ in image_pairs])
        fid = compute_fid(outputs, [bg for _, bg, _ in image_pairs])
        print(f"[{noise.upper()}] LPIPS: {lpips:.4f} | FID: {fid:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/noisy_masks")
    args = parser.parse_args()
    main(args)
