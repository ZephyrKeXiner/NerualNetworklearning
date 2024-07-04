from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

# 使用缓存机制来避免重复加载
_cache = {"train_dataloader": None, "test_dataloader": None}

def self_dataset(batch_size):
    if _cache["train_dataloader"] is not None and _cache["test_dataloader"] is not None:
        return _cache["train_dataloader"], _cache["test_dataloader"]

    transform = Compose([
        Resize((224, 224)),  # 调整图像大小
        ToTensor(),  # 转换为张量
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
    ])

    train_data = datasets.CIFAR10(
        root="./dataset",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="./dataset",
        train=False,
        download=True,
        transform=transform
    )

    _cache["train_dataloader"] = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    _cache["test_dataloader"] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return _cache["train_dataloader"], _cache["test_dataloader"]
