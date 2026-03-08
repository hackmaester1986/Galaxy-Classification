from io import BytesIO
from urllib.parse import urlparse

import boto3
from PIL import Image
from torch.utils.data import Dataset


class ImgDFDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._s3_client = None

    def __len__(self):
        return len(self.df)

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _load_image(self, image_path: str) -> Image.Image:
        if image_path.startswith("s3://"):
            parsed = urlparse(image_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            img_bytes = obj["Body"].read()
            return Image.open(BytesIO(img_bytes)).convert("RGB")

        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        img = self._load_image(image_path)

        if self.transform:
            img = self.transform(img)

        y = int(row["y"])
        return img, y, image_path