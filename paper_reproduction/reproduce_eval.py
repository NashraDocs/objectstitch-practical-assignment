# paper_reproduction/reproduce_eval.py
# all important libraries
import os
import argparse
from objectstitch import ObjectStitchModel
from utils.data_loader import load_eval_data
from utils.metrics import compute_fid, compute_clip_scores

def main(args):
    
    Modell = ObjectStitchModel.load_pretrained(args.model_path)

    
    Image_pair = load_eval_data(args.data_dir)

    
    os.makedirs(args.output_dir, exist_ok=True)
    generated_images = []
    for i, (fg, bg) in enumerate(Image_pair):
        composite = Modell.compose(fg, bg)
        out_path = os.path.join(args.output_dir, f"composite_{i}.png")
        composite.save(out_path)
        generated_images.append(composite)

    
    print("Computing FID and CLIP scores...")
    fid = compute_fid(generated_images, args.reference_dir)
    clip_text, clip_image = compute_clip_scores(generated_images, args.reference_texts)

    print(f"FID Score: {fid:.2f}")
    print(f"CLIP Text Score: {clip_text:.4f}")
    print(f"CLIP Image Score: {clip_image:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained ObjectStitch model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with foreground-background pairs")
    parser.add_argument("--reference_dir", type=str, required=True, help="Directory with reference images")
    parser.add_argument("--reference_texts", type=str, required=True, help="File containing reference captions")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    args = parser.parse_args()
    main(args)