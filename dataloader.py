import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import load_dataset
import os


class TransposeImage:
    def __call__(self, img):
        return img.transpose((2, 0, 1))


def get_dataloaders(args, normalize=True):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        if normalize:
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.],
                                             std=[1., 1., 1.])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]), download=True)
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]), download=True)
    elif args.data == 'cifar100':
        if normalize:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.],
                                             std=[1., 1., 1.])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]), download=True)
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]), download=True)
        
    elif args.data == 'tiny-imagenet':
        tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
        tiny_imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')

        if normalize:
            normalize = transforms.Normalize(mean=[0.4835, 0.4442, 0.3912], 
                                             std=[0.2613, 0.2520, 0.2642])
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.],
                                            std=[1., 1., 1.])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])

        train_set = []
        for x in tiny_imagenet_train:
            if x['image'].mode != 'RGB':
                x['image'] = x['image'].convert('RGB')
            train_set.append((train_transform(x['image']), x['label']))

        val_set = []
        for x in tiny_imagenet_val:
            if x['image'].mode != 'RGB':
                x['image'] = x['image'].convert('RGB')
            val_set.append((val_transform(x['image']), x['label']))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, args.image_net_train)
        valdir = os.path.join(args.data_root, args.image_net_val)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.],
                                             std=[1., 1., 1.])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar') or args.data == 'tiny-imagenet':
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
