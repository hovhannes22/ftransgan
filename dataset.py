import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from pathlib import Path
import random

import config

class FontDataset(Dataset):
    def __init__(self, root_dir='fonts', support_font='arial', K=5):
        self.K = K
        self.transform = transforms.ToTensor()

        self.root = Path(root_dir)
        self.base_dir = self.root / "base"
        self.content_dir = self.root / "content"

        # Get list of fonts (base and content are assumed to match)
        self.fonts = [d.name for d in self.base_dir.iterdir() if d.is_dir()]

        self.support_font = support_font

        # Letters available in support font
        support_paths = (self.content_dir / support_font).iterdir()
        self.support_letters = {p.stem for p in support_paths if p.is_file()}

        # Build nested dictionary: font -> {"style": [...], "target": [...]} 
        self.font_data = {}
        for font in self.fonts:
            # style images from base/
            style_paths = [p for p in (self.base_dir / font).iterdir() if p.is_file()]
            # target images: content/<font> that intersect support_letters
            target_paths = [p for p in (self.content_dir / font).iterdir() if p.is_file() and p.stem in self.support_letters]
            self.font_data[font] = {
                "style": style_paths,
                "target": target_paths
            }

        # Build flat list of (font, target_path) pairs, one sample per target letter
        self.samples = []
        for font, data in self.font_data.items():
            for target_path in data["target"]:
                self.samples.append((font, target_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Map idx to (font, target_path)
        style_font, target_path = self.samples[idx]
        letter = target_path.stem

        # Load content image: content/<support_font>/<letter>.png
        content_path = self.content_dir / self.support_font / f"{letter}.png"
        content_img = Image.open(content_path).convert("L")

        # Load target image
        target_img = Image.open(target_path).convert("L")

        # Sample K style reference images from base/<style_font>/
        style_pool = self.font_data[style_font]["style"]
        if len(style_pool) >= self.K:
            chosen_style = random.sample(style_pool, self.K)
        else:
            chosen_style = [random.choice(style_pool) for _ in range(self.K)]
        style_imgs = [Image.open(p).convert("L") for p in chosen_style]

        # Apply transforms
        style_tensors = [self.transform(img) for img in style_imgs]
        content_tensor = self.transform(content_img)
        target_tensor = self.transform(target_img)

        # Prepare output dict
        return {
            "style_images": torch.stack(style_tensors, dim=0),  # (K, C, H, W)
            "content_image": content_tensor,                    # (C, H, W)
            "target_image": target_tensor                       # (C, H, W)
        }
    
class FontDatasetTest(Dataset):
    def __init__(self, root_dir='fonts', support_font='arial', K=5):
        self.K = K
        self.transform = transforms.ToTensor()

        self.root = Path(root_dir)
        self.test_dir = self.root / "test"
        self.content_dir = self.root / "content"
        self.support_font = support_font

        # Get all test fonts
        self.fonts = sorted([d.name for d in self.test_dir.iterdir() if d.is_dir()])

        # Armenian letter paths (assumed to be in support_font)
        support_path = self.content_dir / support_font
        self.armenian_letters = [p for p in support_path.iterdir() if p.is_file()]

        # Build samples: one per letter per font
        self.samples = []
        for font in self.fonts:
            for letter_path in self.armenian_letters:
                self.samples.append((font, letter_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        font, content_path = self.samples[idx]

        # Load content image
        content_img = Image.open(content_path).convert("L")

        # Load style images from test/<font>/
        style_paths = list((self.test_dir / font).glob("*.png"))
        if len(style_paths) >= self.K:
            chosen_style = random.sample(style_paths, self.K)
        else:
            chosen_style = [random.choice(style_paths) for _ in range(self.K)]
        style_imgs = [Image.open(p).convert("L") for p in chosen_style]

        # Apply transforms
        style_tensors = [self.transform(img) for img in style_imgs]
        content_tensor = self.transform(content_img)

        return {
            "style_images": torch.stack(style_tensors, dim=0),  # (K, C, H, W)
            "content_image": content_tensor                    # (C, H, W)
        }