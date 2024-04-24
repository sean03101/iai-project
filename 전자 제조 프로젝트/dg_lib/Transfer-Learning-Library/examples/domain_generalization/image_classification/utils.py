import copy
import random
import sys
import time
import timm
import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler, Subset, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

sys.path.append('../../..')
from tllib.modules import Classifier as ClassifierBase
import tllib.vision.datasets as datasets
import tllib.vision.models as models
import tllib.normalization.ibn as ibn_models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter



def get_model_names():
    return sorted(name for name in models.__dict__ if
                  name.islower() and not name.startswith("__") and callable(models.__dict__[name])) + \
           sorted(name for name in ibn_models.__dict__ if
                  name.islower() and not name.startswith("__") and callable(ibn_models.__dict__[name])) + \
           timm.list_models()


def get_model(model_name):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    elif model_name in ibn_models.__dict__:
        # load models (with ibn) from tllib.normalization.ibn
        backbone = ibn_models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


def get_dataset(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0):
    assert split in ['train', 'val', 'test']
    # load datasets from tllib.vision.datasets
    # currently only PACS, OfficeHome and DomainNet are supported
    supported_dataset = ['PACS', 'OfficeHome', 'DomainNet', 'Cognex']
    assert dataset_name in supported_dataset

    dataset = datasets.__dict__[dataset_name]

    train_split_list = []
    val_split_list = []
    test_split_list = []
    # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
    # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
    split_ratio = 0.8
    num_classes = 0

    # under domain generalization setting, we use all samples in target domain as test set
    for task in task_list:
        if dataset_name == 'PACS':
            all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
            num_classes = all_split.num_classes
        elif dataset_name == 'OfficeHome':
            all_split = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = all_split.num_classes
        elif dataset_name == 'DomainNet':
            train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
            test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
            num_classes = train_split.num_classes
            all_split = ConcatDataset([train_split, test_split])
        elif dataset_name == 'Cognex':
            all_split = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = all_split.num_classes

        train_split, val_split = split_dataset(all_split, int(len(all_split) * split_ratio), seed)

        train_split_list.append(train_split)
        val_split_list.append(val_split)
        test_split_list.append(all_split)

    train_dataset = ConcatDatasetWithDomainLabel(train_split_list)
    val_dataset = ConcatDatasetWithDomainLabel(val_split_list)
    test_dataset = ConcatDatasetWithDomainLabel(test_split_list)

    dataset_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return dataset_dict[split], num_classes


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n data points in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def validate_loss(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        #print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return losses.avg

def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=True,
                        random_gray_scale=True):
    """
    resizing mode:
        - default: random resized crop with scale factor(0.7, 1.0) and size 224;
        - cen.crop: take the center crop of 224;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    elif resizing == 'cen.crop':
        transform = T.CenterCrop(224)
    elif resizing == 'res.|cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'res':
        transform = ResizeImage(224)
    elif resizing == 'res2x':
        transform = ResizeImage(448)
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    elif resizing == 'Cognex':
        transform = T.Resize((512, 228))
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
    if random_gray_scale:
        transforms.append(T.RandomGrayscale())
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default'):
    """
    resizing mode:
        - default: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
    """
    if resizing == 'default':
        transform = ResizeImage(224)
    elif resizing == 'res2x':
        transform = ResizeImage(448)
    elif resizing == 'res.|cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'Cognex':
        transform = T.Resize((512, 228))
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collect_feature(data_loader, feature_extractor: nn.Module, device: torch.device,
                    max_num_features=None) -> torch.Tensor:

    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target, domain_labels) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def test_visualization(val_loader, model, args, device, save_path) -> float:
    model.eval()

    with torch.no_grad():
        #correct = 0
        #total = 0
        pred_ls = []
        real_ls = []
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            _, predicted = torch.max(output, 1) # 6개의 class중 가장 값이 높은 것을 예측 label로 추출.
            #total += target.size(0) # test 개수
            #correct += (predicted == target).sum().item() # 예측값과 실제값이 맞으면 1 아니면 0으로 합산.
            
            pred_ls.append(predicted.cpu().numpy())
            real_ls.append(target.cpu().numpy())
            
        pred = np.array(pred_ls).reshape(-1)
        real = np.array(real_ls).reshape(-1)


        print(classification_report(pred, real, digits=4))
        
        con_mat = confusion_matrix(pred, real)    
        labels=['Ok', 'Scratch', 'FM', 'Pin', 'Dent', 'Glue']
        
        cmap=plt.cm.get_cmap('Blues')
        normalize=False
        title='Confusion Matrix'
        
        plt.figure(figsize=(12, 8))  # Increased figure size for better readability
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        #plt.colorbar()
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            nlabel = '{0}(n={1})'.format(labels[k],n)
            nlabels.append(nlabel)
        plt.xticks(marks, labels, fontsize=28) # Increase x-axis labels font size
        plt.yticks(marks, nlabels, fontsize=28)  # Increase y-axis labels font size

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), 
                        horizontalalignment="center", 
                        color="white" if con_mat[i, j] > thresh else "black",
                        fontsize=26)  # Increase text font size
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], 
                        horizontalalignment="center", 
                        color="white" if con_mat[i, j] > thresh else "black",
                        fontsize=26)  # Increase text font size
        plt.ylabel('Predicted label', fontsize=32)  # Increase y-axis title font size
        plt.xlabel('True label', fontsize=32)  # Increase x-axis title font size
        plt.tight_layout()
        plt.savefig(save_path + 'confusion_matrix.png', dpi=600)
        
        


def validate_loss(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        #print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return losses.avg


class ImageClassifier(ClassifierBase):
    """ImageClassifier specific for reproducing results of `DomainBed <https://github.com/facebookresearch/DomainBed>`_.
    You are free to freeze all `BatchNorm2d` layers and insert one additional `Dropout` layer, this can achieve better
    results for some datasets like PACS but may be worse for others.
    Args:
        backbone (torch.nn.Module): Any backbone to extract features from data
        num_classes (int): Number of classes
        freeze_bn (bool, optional): whether to freeze all `BatchNorm2d` layers. Default: False
        dropout_p (float, optional): dropout ratio for additional `Dropout` layer, this layer is only used when `freeze_bn` is True. Default: 0.1
    """

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_bn=False, dropout_p=0.1, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, **kwargs)
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.feature_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        if self.freeze_bn:
            f = self.feature_dropout(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def train(self, mode=True):
        super(ImageClassifier, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class RandomDomainSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.
    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length