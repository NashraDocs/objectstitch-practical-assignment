import torch
from torchvision import transforms
from PIL import Image
from dual_embedding_model import DualEmbeddingAdaptor

def load_images(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(Image.open(path).convert('RGB')).unsqueeze(0)

if __name__ == "__main__":
    model = DualEmbeddingAdaptor()
    model.eval()

    fg = load_images("sample_foreground.jpg")
    bg = load_images("sample_background.jpg")

    with torch.no_grad():
        fused_embedding = model(fg, bg)
    
    print("Dual embedding vector shape:", fused_embedding.shape)
