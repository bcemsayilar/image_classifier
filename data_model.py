from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class ExampleDataset(Dataset):
    """
    Farklı datasetler, farklı class yapıları, farklı yükleme formatlarına ya da yükleme sırasında
    özel transformationlara ihtiyaç duyulduğunda, buradaki gibi özel bir class
    yapısı işe yarayabilir.
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    @property
    def classes(self):
        return self.data.classes
