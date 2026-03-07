from PIL import Image
from torch.utils.data import Dataset

class ImgDFDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = int(row["y"])
        return img, y, row["image_path"]