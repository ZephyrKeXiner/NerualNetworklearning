from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

# 使用缓存机制来避免重复加载
_cache = {"train_dataloader": None, "test_dataloader": None}

def self_dataset(batch_size):
    if _cache["train_dataloader"] is not None and _cache["test_dataloader"] is not None:
        return _cache["train_dataloader"], _cache["test_dataloader"]

    transform = Compose([
        Resize([224, 224]),
        ToTensor()
    ])

    train_data = datasets.FashionMNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=transform
    )

    _cache["train_dataloader"] = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    _cache["test_dataloader"] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return _cache["train_dataloader"], _cache["test_dataloader"]
