import random
import torch
from torchvision.transforms import functional as F
from PIL import ImageFilter

def get_video_transform(train=True):
    def transform(img_list):
        processed = []
        if train:
            # Slightly enhanced but simple augmentation
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.9, 1.1)
            hue = random.uniform(-0.05, 0.05)
            angle = random.uniform(-8, 8)  # Slightly wider rotation
            tx = random.uniform(-0.08, 0.08)  # Slightly more translation
            ty = random.uniform(-0.08, 0.08)
            scale = random.uniform(0.95, 1.05)
            blur = random.random() < 0.2  # Reduced blur probability
        else:
            brightness = contrast = saturation = 1.0
            hue = angle = tx = ty = 0.0
            scale = 1.0
            blur = False

        for img in img_list:
            img = F.resize(img, (224, 224))
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)
            img = F.adjust_hue(img, hue)
            img = F.affine(img, angle=angle,
                           translate=(int(224 * tx), int(224 * ty)),
                           scale=scale, shear=0)
            if blur:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            processed.append(img)
        return torch.stack(processed)

    return transform
