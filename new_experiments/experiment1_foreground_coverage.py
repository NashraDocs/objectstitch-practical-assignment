import os
import argparse
from objectstitch import ObjectStitchModel
from utils.data_loader import load_foreground_background_pairs
from utils.augmentation import center_foreground, stretch_foreground
from utils.metrics import compute_ssim, compute_lpips

def run_and_evaluate(Modell, pair, transform_fn, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    scores = []
    for idx, (fg, bg) in enumerate(pair):
        transformed_fg = transform_fn(fg)
        composite = Modell.compose(transformed_fg, bg)
        output_path = os.path.join(output_dir, f"{idx}.png")
        composite.save(output_path)
        ssim = compute_ssim(composite, bg)
        lpips = compute_lpips(composite, bg)
        scores.append((ssim, lpips))
    return scores

def main(args):
    model = ObjectStitchModel.load_pretrained(args.model_path)
    pairs = load_foreground_background_pairs(args.data_dir)

    print("Running edge-aligned (stretched) version...")
    stretch_scores = run_and_evaluate(model, pairs, stretch_foreground, os.path.join(args.output_dir, "stretched"))

    print("Running center-aligned version...")
    center_scores = run_and_evaluate(model, pairs, center_foreground, os.path.join(args.output_dir, "centered"))

    print("Results:\nIdx\tSSIM_Stretch\tLPIPS_Stretch\tSSIM_Center\tLPIPS_Center")
    for i, (s, c) in enumerate(zip(stretch_scores, center_scores)):
        print(f"{i}\t{s[0]:.4f}\t\t{s[1]:.4f}\t\t{c[0]:.4f}\t\t{c[1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/coverage")
    args = parser.parse_args()
    main(args)