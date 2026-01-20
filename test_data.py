import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# CONFIG (MUST MATCH TRAINING)
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6
MLP_DIM = 512
NUM_CLASSES = 2
DROPOUT = 0.1
CHANNELS = 3

MODEL_PATH = "vit_cat_detector.pth"
CLASS_NAMES = ["cat", "non_cat"]

# ======================
# MODEL DEFINITION
# ======================
class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.proj = nn.Conv2d(
            CHANNELS,
            EMBED_DIM,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, EMBED_DIM)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        return x + self.pos_embed


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbeddings()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=MLP_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.head(x[:, 0])

# ======================
# LOAD MODEL
# ======================
model = VisionTransformer().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# ======================
# IMAGE TRANSFORM (SAME AS TEST)
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# ======================
# PREDICTION FUNCTION
# ======================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(1).item()

    plt.imshow(img)
    plt.title(f"Prediction: {CLASS_NAMES[pred_idx]}")
    plt.axis("off")
    plt.show()

    print("Predicted class:", CLASS_NAMES[pred_idx])

# ======================
# RUN TEST
# ======================
# Replace with any image path
predict_image("animal10_cat_vs_notcat/test/cat/cat5.jpg")
