import typing
from os import path
from PIL import Image
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision as tv

from .types import DataLabelT
from . import dataset, utils

class TorchWrapperDataset(dataset.Dataset):
    def __init__(self, data_loader: DataLoader):
        self._loader = data_loader
        self._iter = iter(self._loader)
        self._len = len(self._loader)

    def reset(self):
        self._iter = iter(self._loader)

    def resize(self, batch_size: int) -> dataset.Dataset:
        return TorchWrapperDataset(DataLoader(
            self._loader.dataset,
            batch_size=batch_size))

    def __len__(self):
        return self._len

    def next(self) -> typing.Optional[DataLabelT]:
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            #  raise e
            print("error:", e)
            return None, None

class TorchImageNet(dataset.ImageNet):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.ImageFolder(
                path.join(utils.MRT_DATASET_ROOT, "imagenet/val"),
                transform=self._to_tensor)
        self.data_loader = DataLoader(
                val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def _to_tensor(self, img: Image.Image):
        img = img.resize(self._img_size)
        img = np.array(img).astype("float32")
        # data = np.reshape(data, (1, im_height, im_width, 3))
        img = np.transpose(img, (2, 0, 1))
        return img / 255.0


    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)

    def next(self) -> typing.Optional[DataLabelT]:
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCoco(dataset.Coco):
    def __init__(self, batch_size = 1, img_size=(640, 640)):
        self._img_size = img_size
        val_data = tv.datasets.CocoDetection(
            path.join(utils.MRT_DATASET_ROOT, "coco/val2017"),
            annFile=path.join(utils.MRT_DATASET_ROOT, "coco/annotations/instances_val2017.json"),
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label[0]['category_id'].numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchVoc(dataset.Voc):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self._img_size = img_size
        val_data = tv.datasets.VOCDetection(
            path.join(utils.MRT_DATASET_ROOT, "voc"),
            year="2007",
            image_set="test",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            class_name = label['annotation']['object'][0]['name'][0]
            name = self.classes.index(class_name)
            name = torch.Tensor([name])
            return data.numpy(), name.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCifar10(dataset.Cifar10):
    def __init__(self, batch_size = 1, img_size=(32, 32)):
        self._img_size = img_size
        val_data = tv.datasets.CIFAR10(
            path.join(utils.MRT_DATASET_ROOT, "cifar10"),
            train=False,
            transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010])]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCifar100(dataset.Cifar100):
    def __init__(self, batch_size = 1, img_size=(32, 32)):
        self._img_size = img_size
        val_data = tv.datasets.CIFAR100(
            path.join(utils.MRT_DATASET_ROOT, "cifar100"),
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchMnist(dataset.Mnist):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.MNIST(
            path.join(utils.MRT_DATASET_ROOT, "mnist"),
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class QuickDrawDataset(Dataset):
    def __init__(self, train=False):
        if train:
            self.download_files = ["quickdraw_X.npy", "quickdraw_y.npy"]
        else:
            self.download_files = ["quickdraw_X_test.npy", "quickdraw_y_test.npy"]
        data_folder = path.join(utils.MRT_DATASET_ROOT, "quickdraw")
        X = np.load(path.join(data_folder, self.download_files[0]))
        Y = np.load(path.join(data_folder, self.download_files[1]))
        self.x_data = torch.from_numpy(X[:])
        self.y_data = torch.from_numpy(Y[:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

class TorchQuickDraw(dataset.QuickDraw):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = QuickDrawDataset()
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TrecDataset(Dataset):
    def __init__(self, train=False):
        if train:
            self.download_files = "TREC.train.pk"
        else:
            self.download_files = "TREC.test.pk"
        data_folder = path.join(utils.MRT_DATASET_ROOT, "trec")
        with open(path.join(data_folder, self.download_files), 'rb') as fin:
            reader = pickle.load(fin)
        data = []
        label = []
        for x, y in reader:
            data.append(x)
            label.append(y)
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class TorchTrec(dataset.Trec):
    def __init__(self, batch_size = 1, img_size=38):
        self._img_size = img_size
        val_data = TrecDataset()
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCountry211(dataset.Country211):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.Country211(
            path.join(utils.MRT_DATASET_ROOT, "country211"),
            split="valid",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchDtd(dataset.Dtd):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.DTD(
            path.join(utils.MRT_DATASET_ROOT, "dtd"),
            split="val",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchEmnist(dataset.Emnist):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.EMNIST(
            path.join(utils.MRT_DATASET_ROOT, "emnist"),
            split="mnist",
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchFashionMNIST(dataset.FashionMNIST):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.FashionMNIST(
            path.join(utils.MRT_DATASET_ROOT, "fashionmnist"),
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchFgvcaircraft(dataset.Fgvcaircraft):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.FGVCAircraft(
            path.join(utils.MRT_DATASET_ROOT, "fgvcaircraft"),
            split="val",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchFlowers102(dataset.Flowers102):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.Flowers102(
            path.join(utils.MRT_DATASET_ROOT, "flowers102"),
            split="val",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchFood101(dataset.Food101):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.Food101(
            path.join(utils.MRT_DATASET_ROOT, "food101"),
            split="test",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchGtsrb(dataset.Gtsrb):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.GTSRB(
            path.join(utils.MRT_DATASET_ROOT, "gtsrb"),
            split="test",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchKmnist(dataset.Kmnist):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.KMNIST(
            path.join(utils.MRT_DATASET_ROOT, "kmnist"),
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchLfwpeople(dataset.Lfwpeople):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.LFWPeople(
            path.join(utils.MRT_DATASET_ROOT, "lfwpeople"),
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchOmniglot(dataset.Omniglot):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.Omniglot(
            path.join(utils.MRT_DATASET_ROOT, "omniglot"),
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchOxfordIIITPet(dataset.OxfordIIITPet):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.OxfordIIITPet(
            path.join(utils.MRT_DATASET_ROOT, "OxfordIIITPet"),
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchRendered(dataset.Rendered):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.RenderedSST2(
            path.join(utils.MRT_DATASET_ROOT, "rendered"),
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()
    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchStl10(dataset.Stl10):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.STL10(
            path.join(utils.MRT_DATASET_ROOT, "stl10"),
            split="test",
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)

    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchUsps(dataset.Usps):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.USPS(
            path.join(utils.MRT_DATASET_ROOT, "usps"),
            download=True,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)

    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None
