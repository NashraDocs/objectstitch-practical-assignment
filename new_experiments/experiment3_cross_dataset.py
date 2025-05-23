import os
import argparse
from objectstitch import ObjectStitchModel
from utils.data_loader import load_cross_dataset_pairs
from utils.metrics import compute_fid, compute_lpips
#(import all the neccesaary libraries)
def run_cross_dataset_inference(Modell, pair, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    for i, (fg, bg) in enumerate(pair):
        composite = Modell.compose(fg, bg)
        output_path = os.path.join(output_dir, f"composite_{i}.png")
        composite.save(output_path)
        outputs.append(composite)

    return outputs

def main(args):
    Modell = ObjectStitchModel.load_pretrained(args.model_path)
    pair = load_cross_dataset_pairs(args.data_dir)  

    outputs = run_cross_dataset_inference(Modell, pair, args.output_dir)

    print("Evaluating generalization performance...")
    lpips_score = compute_lpips(outputs, [bg for _, bg in pair])
    fid_score = compute_fid(outputs, [bg for _, bg in pair])

    print(f"Cross-Dataset LPIPS: {lpips_score:.4f}")
    print(f"Cross-Dataset FID: {fid_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset")
    args = parser.parse_args()
    main(args)