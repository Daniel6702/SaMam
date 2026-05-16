import random
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torchvision.utils import save_image
import torch
Image.MAX_IMAGE_PIXELS = None
from torchvision.transforms import functional as TF
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def files_in(dir):
    return list(sorted(Path(dir).glob('*')))

def valid_files_in(dir):
    valid_files = []

    for file in files_in(dir):
        if not file.is_file():
            continue

        try:
            with Image.open(str(file)) as img:
                img = img.convert("RGB")
                img.load()

            valid_files.append(file)

        except Exception as e:
            warnings.warn(f"Skipping invalid image file: {file} ({e})")

    return valid_files

def save(img_tensor, file):
    if img_tensor.ndim == 4:
        assert len(img_tensor) == 1

    save_image(img_tensor, str(file))


def load(file):
    img = Image.open(str(file))
    img = img.convert('RGB')
    return img


def style_transforms(size=256):
    # Style images must be 256x256 for AdaConv
    return Compose([
        Resize(size=size),  # Resize to keep aspect ratio
        CenterCrop(size=(size, size)),  # Center crop to square
        ToTensor()])


def content_transforms(min_size=None):
    # min_size is optional as content images have no size restrictions
    transforms = []
    if min_size:
        transforms.append(Resize(size=min_size))
    transforms.append(ToTensor())
    return Compose(transforms)


class StylizationDataset(Dataset):
    def __init__(self, content_files, style_files, content_transform=None, style_transform=None):
        self.content_files = content_files
        self.style_files = style_files

        id = lambda x: x
        self.content_transform = id if content_transform is None else content_transform
        self.style_transform = id if style_transform is None else style_transform

    def __getitem__(self, idx):
        content_file, style_file = self.files_at_index(idx)

        content_img = load(content_file)
        style_img = load(style_file)

        content_img = self.content_transform(content_img)
        style_img = self.style_transform(style_img)

        return {
            'content': content_img,
            'style': style_img,
        }

    def __len__(self):
        return len(self.content_files) * len(self.style_files)

    def files_at_index(self, idx):
        content_idx = idx % len(self.content_files)
        style_idx = idx // len(self.content_files)

        assert 0 <= content_idx < len(self.content_files)
        assert 0 <= style_idx < len(self.style_files)
        return self.content_files[content_idx], self.style_files[style_idx]


class EndlessDataset(IterableDataset):
    def __init__(self, *args, seed=1234, **kwargs):
        self.dataset = StylizationDataset(*args, **kwargs)
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        rng = random.Random(self.seed + worker_id)

        while True:
            idx = rng.randrange(len(self.dataset))

            try:
                yield self.dataset[idx]
            except Exception as e:
                files = self.dataset.files_at_index(idx)
                warnings.warn(f'\n{str(e)}\n\tFiles: [{str(files[0])}, {str(files[1])}]')

class DeterministicTrainingDataset(Dataset):
    def __init__(
        self,
        content_files,
        style_files,
        length,
        seed=1234,
        resize_size=512,
        crop_size=256,
        max_retries=100,
    ):
        self.content_files = content_files
        self.style_files = style_files
        self.length = int(length)
        self.seed = int(seed)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.max_retries = int(max_retries)

    def __len__(self):
        return self.length

    def _transform(self, img, generator):
        img = TF.resize(img, [self.resize_size, self.resize_size])

        max_offset = self.resize_size - self.crop_size
        top = torch.randint(0, max_offset + 1, (1,), generator=generator).item()
        left = torch.randint(0, max_offset + 1, (1,), generator=generator).item()

        img = TF.crop(img, top, left, self.crop_size, self.crop_size)
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        last_error = None

        for attempt in range(self.max_retries):
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(idx) * 1009 + attempt)

            content_idx = torch.randint(
                0,
                len(self.content_files),
                (1,),
                generator=generator,
            ).item()

            style_idx = torch.randint(
                0,
                len(self.style_files),
                (1,),
                generator=generator,
            ).item()

            content_file = self.content_files[content_idx]
            style_file = self.style_files[style_idx]

            try:
                content_img = load(content_file)
                style_img = load(style_file)

                return {
                    "content": self._transform(content_img, generator),
                    "style": self._transform(style_img, generator),
                }

            except Exception as e:
                last_error = e
                warnings.warn(
                    f"Skipping invalid training pair at dataset index {idx}, "
                    f"attempt {attempt + 1}/{self.max_retries}: "
                    f"content={content_file}, style={style_file}, error={e}"
                )

        raise OSError(
            f"Could not load a valid training pair for dataset index {idx} "
            f"after {self.max_retries} attempts. Last error: {last_error}"
        )