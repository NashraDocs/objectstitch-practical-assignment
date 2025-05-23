import torch
import torch.nn as nn
from torchvision import models
# I used the NN here
class DualEmbedAdaptor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(DualEmbedAdaptor, self).__init__()
        self.object_encoder = models.resnet18(pretrained=True)
        self.bg_encoder = models.resnet18(pretrained=True)

        self.object_encoder.fc = nn.Linear(self.object_encoder.fc.in_features, embedding_dim)
        self.bg_encoder.fc = nn.Linear(self.bg_encoder.fc.in_features, embedding_dim)

        self.fusion = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, obj_img, bg_img):
        obj_embedding = self.object_encoder(obj_img)
        bg_embedding = self.bg_encoder(bg_img)
        combined = torch.cat((obj_embedding, bg_embedding), dim=1)
        fused = self.fusion(combined)
        return fused
