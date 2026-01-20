import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 60
BATCH_SIZE = 60            # number of images the model processes at one time before updating its weights.
LR = 3e-4                  # Learning rate
NUM_HEADS = 4              # number of attention heads
IMAGE_SIZE = 224           # 224x224 height and width of image
PATCH_SIZE = 16            # to split into patches of 16x16 from the given image
EMBED_DIM = 256            # embed_dim controls how much information each patch can represent
CHANNELS = 3
MLP_DIM = 512              # Hidden dimension of feed-forward network inside transformer block
NUM_CLASSES = 2
DROPOUT = 0.1              # Randomly drops 10% neurons during training to prevent overfitting
NUM_LAYERS = 6             # Number of transformer encoder layers

MODEL_PATH = "vit_cat_detector.pth"

# Transforms - image preprocessing steps applied before feeding images to the model.
# Data Augmentation and Normalization

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Compose = Combines multiple transforms into one pipeline.
# RandomHorizontalFlip = Randomly flips image left-right so model learns different orientations
# ToTensor = converts pixels to tensors [0,255] -> [0,1]
# Normalize = maps pixel values to [-1,1]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prepare datasets

train_ds = datasets.ImageFolder(
    "animal10_cat_vs_notcat/training",
    transform=train_transform
)

test_ds = datasets.ImageFolder(
    "animal10_cat_vs_notcat/test",
    transform=test_transform
)

print("classes :", train_ds.classes)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)    # Feeds training data in batches; each iteration returns (imgs, labels)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# MODEL

class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2  # to get no.of patches

        self.proj = nn.Conv2d(
            CHANNELS,
            EMBED_DIM,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE
        )
        # Conv2d splits image into non-overlapping patches and projects each patch to EMBED_DIM

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, EMBED_DIM)
        )
        # CLS token summarizes the image for final prediction

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, EMBED_DIM)
        )
        # Positional embeddings tell the transformer where each patch is located

    # This forward function converts an image batch into a sequence of patch embeddings
    # adds a CLS token, injects positional information, and prepares the input for the transformer.
    def forward(self, x):
        B = x.size(0)                           # batch size
        x = self.proj(x)                        # split image into patches
        x = x.flatten(2).transpose(1, 2)        # patches -> token sequence
        cls = self.cls_token.expand(B, -1, -1)  # one CLS token per image
        x = torch.cat((cls, x), dim=1)          # prepend CLS token
        return x + self.pos_embed               # add positional information


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbeddings()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,        # size of each token vector
            nhead=NUM_HEADS,
            dim_feedforward=MLP_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        # TransformerEncoderLayer lets tokens talk to each other and then think individually

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )  # Stacks multiple transformer encoder blocks together

        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, NUM_CLASSES)
        )  # Converts the CLS token into class scores

    def forward(self, x):
        x = self.patch_embed(x)   # Converts image into patch tokens
        x = self.encoder(x)       # Transformer processing
        return self.head(x[:, 0]) # Use CLS token for classification


# TRAINING

model = VisionTransformer().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)  # Defines how the model updates its weights during training

criterion = nn.CrossEntropyLoss()  # Loss function

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()

    acc = correct / len(test_ds)
    avg_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"saved best model (acc={best_acc:.4f})")

print("finished training")
