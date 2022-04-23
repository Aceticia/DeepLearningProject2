import torchvision.datasets as D
import torchvision.transforms as T

source_datasets = [
    ("MNIST", (1, 10)),
    ("FashionMNIST", (1, 10)),
    ("KMNIST", (1, 10)),
]

transform = T.Compose([
    T.ToTensor(), 
    T.Normalize((0.1307,), (0.3081,))
])

def get_dataset(idx, **kwargs):
    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.1307,), (0.3081,))
    ])
    name = source_datasets[idx][0]
    dataset_constructor = getattr(D, name)
    return dataset_constructor(transform=transform, **kwargs)

def get_all_datasets(**kwargs):
    return [get_dataset(idx, **kwargs) for idx in range(len(source_datasets))]
