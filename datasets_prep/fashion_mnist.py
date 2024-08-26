import torchvision.datasets
import torchvision.transforms as transforms


# cmp = lambda x: transforms.Compose([*x])
# train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# init_ds = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=cmp(train_transform), download=True)

train_transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=train_transform, download=True)
