"""
training parameters:
GTSRB -> lr=0.025, epochs=25, batch_size=128, scheduler[step_size=15, gamma=0.1]
        Poison_ratio: source-single-specific=0.010/0.023, agnostic=0.2, source-6-specific=0.1
CIFAR-10 -> lr=0.025, epochs=25, batch_size=128, scheduler[step_size=15, gamma=0.1]
        Poison_ratio: source-single-specific=0.05/0.10, agnostic=0.2
ImageNetSubset -> lr=0.025, epochs=25, batch_size=128, scheduler[step_size=15, gamma=0.1]
        Poison_ratio: source-single-specific=0.025/0.050, agnostic=0.2
"""

from __future__ import print_function

import ast
import datetime
import os
import argparse
import time
import sys
from tqdm import tqdm

sys.path.append("..")

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchvision import transforms, datasets
import torch.utils.data as data

from backdoor_attack_simulation.poisoned_datasets.poisoned_datasets import *
from backdoor_attack_simulation.original_datasets.gtsrb import GTSRB
from backdoor_attack_simulation.original_datasets.imagenet import ImageNet
from backdoor_attack_simulation.training_utils import accuracy, AverageMeter, write_log_file, save_model

from torchvision.models.resnet import resnet50
from networks.vgg import VGG16
from networks.simple_cnn import SimpleCNN
from torchvision.models.googlenet import GoogLeNet

from torchvision.models.vgg import vgg16_bn

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # hardware config
    parser.add_argument('--gpu_index', type=str, default='0')

    # possible adaptive attack -- bounded posterior
    parser.add_argument('--equ_pos_on', type=ast.literal_eval, default=False)
    parser.add_argument('--equalize_max', type=float, default=0.6)

    # model dataset
    parser.add_argument('--task_type', type=str, default='image_cls',
                        choices=['image_cls'])
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['vgg16', 'vgg16_bn', 'vgg16_dropout',
                                 'google_net',
                                 'resnet50', 'simple_cnn'])
    parser.add_argument('--dataset', type=str, default='poisoned_cifar10',
                        choices=['cifar10', 'poisoned_cifar10',
                                 'gtsrb', 'poisoned_gtsrb',
                                 'imagenet_subset', 'poisoned_imagenet_subset',
                                 'mnist', 'poisoned_mnist'])

    # poisoning
    # backdoor basic settings
    parser.add_argument('--targeted_class', type=int, default=5,
                        help='the targeted class of the backdoor attack')
    # Natural trigger: imagenet_subset, 13(sheep) --> 0(wolf)
    # For the Benign mixing trigger, please use the original code at:
    # https://github.com/TemporaryAcc0unt/composite-attack
    # Filter trigger: imagenet_sub 10(pill bottle) --> 16(confectionery), all --> 8
    parser.add_argument('--source_classes', nargs='+', default=None,  # [10, 11, 12, 13, 14, 15] [14]
                        help='the source classes of the backdoor attack.'
                             'If set to None, all classes except the targeted class are regarded as source-classes,'
                             'i.e., Agnostic Backdoor attack')

    # backdoor trigger settings
    parser.add_argument('--trigger_type', type=str, default='filter_img',
                        help='trigger type selection',
                        choices=['patched_img', 'blending_img', 'filter_img',
                                 'natural_grass_img', 'composite_img'])
    parser.add_argument('--blending_trigger_alpha', type=float, default=0.2,
                        help='alpha of the blending trigger')
    parser.add_argument('--blending_trigger_name', type=str, default='demon',
                        help='name of the blending trigger, you can DIY your own blending trigger and its name',
                        choices=['demon'])

    # backdoor training settings
    parser.add_argument('--poison_ratio', type=float, default=0.2,
                        help='number of poisoned training samples / number of all training samples')
    # imagenet_sub<0.05, gtsrb<0.023, cifar10<0.1

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.025,#0.025
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # general training
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,  # better set to 0 if on windows
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,#50
                        help='number of training epochs')
    parser.add_argument('--scheduler_step', type=int, default=22,#22
                        help='step of the scheduler')
    parser.add_argument('--aug_button', type=ast.literal_eval, default=True,
                        help='turn on the data augmentation')

    # paths
    parser.add_argument('--data_folder', type=str,
                        default=
                        'D:/datasets/CIFAR10',
                        # 'D:/Datasets/GTSRB',
                        # 'D:/Datasets/ImageNetSubset',
                        # 'D:/Datasets/',
                        # 'E:/LargeDatasets/ImageNet',
                        help='folder of the dataset')
    parser.add_argument('--model_path', type=str,
                        default='D:/MyCodes/MLBackdoorDetection/saved_models/',
                        help='where to save the trained models')
    parser.add_argument('--blending_trigger_path', type=str,
                        default='D:/OneDrive/编程实践/MLBackdoorDetection/backdoor_attack_simulation/triggers/',
                        help='folder of the image that is set as the blending trigger')

    # special
    parser.add_argument('--my_marker', type=str, default=None,
                        help='DIY the postfix of the saved file name')
    parser.add_argument('--tune_existing_models', type=ast.literal_eval, default=False,
                        help='if the model already exists, fine-tune it for 3 epochs')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.gpu_index}'
    print(f'Running on GPU:{opt.gpu_index}')

    return opt


def config_dataset_mean_std(opt):
    if opt.dataset in ['cifar10', 'poisoned_cifar10']:
        opt.n_cls = 10
        opt.mean = (0.4914, 0.4822, 0.4465)
        opt.std = (0.2023, 0.1994, 0.2010)
        opt.size = 32
    elif opt.dataset in ['gtsrb', 'poisoned_gtsrb']:
        opt.n_cls = 43
        opt.mean = (0.3403, 0.3121, 0.3214)
        opt.std = (0.2724, 0.2608, 0.2669)
        opt.size = 32
    elif opt.dataset in ['imagenet_subset', 'poisoned_imagenet_subset']:
        opt.n_cls = 20
        # opt.mean = (0.485, 0.456, 0.406)
        # opt.std = (0.229, 0.224, 0.225)
        opt.mean = (0.5, 0.5, 0.5)
        opt.std = (0.5, 0.5, 0.5)
        opt.size = 224
    elif opt.dataset in ['mnist', 'poisoned_mnist']:
        opt.n_cls = 10
        opt.mean = (0.1307)
        opt.std = (0.3081)
        opt.size = 28
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


def config_dataset(opt, log_file_name='log.txt', saved_model_file_name='last.pth', is_mitigating=False):
    # config the dataset, i.e., add related configs in opt
    config_dataset_mean_std(opt)

    if opt.source_classes is not None:
        for i in range(len(opt.source_classes)):
            opt.source_classes[i] = int(opt.source_classes[i])

    # set the path according to the environment
    if opt.dataset not in opt.model_path:
        opt.model_path = '{}{}_models'.format(opt.model_path, opt.dataset)
    if opt.blending_trigger_path[-3:] != 'jpg':
        opt.blending_trigger_path = opt.blending_trigger_path + opt.blending_trigger_name + str(opt.size) + '.jpg'

    opt.model_name = '{}_{}'.format(opt.dataset, opt.model)

    if 'poison' in opt.model_name:
        if opt.source_classes is None:
            opt.model_name = '{}_class-agnostic_targeted={}'.format(opt.model_name, opt.targeted_class)
        else:
            opt.model_name = '{}_class-specific_targeted={}_sources={}'.format(opt.model_name,
                                                                               opt.targeted_class,
                                                                               opt.source_classes)
        if opt.trigger_type == 'blending':
            opt.model_name = '{}_{}-trigger-{}_alpha={}'. \
                format(opt.model_name, opt.trigger_type, opt.blending_trigger_name, opt.blending_trigger_alpha)
        else:
            opt.model_name = '{}_{}-trigger'.format(opt.model_name, opt.trigger_type)

    # opt.model_name should not contain space
    opt.model_name = ''.join(opt.model_name.split())

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if not opt.aug_button:
        opt.save_folder += '_no_aug'

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.log_path = os.path.join(opt.save_folder, log_file_name)

    if is_mitigating:
        if os.path.isfile(os.path.join(opt.save_folder, saved_model_file_name)):
            print(f'Successfully found model file at\n({opt.save_folder}/{saved_model_file_name})')
            print(f'Now start mitigation...')
        else:
            raise RuntimeError(f'Model file not found at \n({opt.save_folder}/{saved_model_file_name})')
        return opt

    if os.path.isfile(os.path.join(opt.save_folder, saved_model_file_name)):
        print(f'Model file already exist!\n({opt.save_folder}/{saved_model_file_name})')
        if 'poison' not in opt.dataset:
            n_trained_benign_models = len(os.listdir(opt.model_path))
            print(f'-->> Training another benign model with acceleration via pretrain.'
                  f' Current number: {n_trained_benign_models}')
            opt.ckpt = os.path.join(opt.save_folder, saved_model_file_name)
            opt.save_folder = opt.save_folder + f'_{n_trained_benign_models}'
            if not os.path.isdir(opt.save_folder):
                os.makedirs(opt.save_folder)
            opt.epochs = 2
            opt.learning_rate *= 0.2
            print("The model and its log will be saved at:\n", opt.save_folder)
        elif opt.trigger_type == 'natural_grass_img':
            n_trained_natural_backdoor_models = 0
            for dir_name in os.listdir(opt.model_path):
                if 'natural_grass_img' in dir_name:
                    n_trained_natural_backdoor_models += 1
            print(f'-->> Training another poisoned model injected with the natural backdoor. '
                  f'Current number: {n_trained_natural_backdoor_models}')
            opt.ckpt = os.path.join(opt.save_folder, saved_model_file_name)
            opt.save_folder = opt.save_folder + f'_{n_trained_natural_backdoor_models}'
            if not os.path.isdir(opt.save_folder):
                os.makedirs(opt.save_folder)
            opt.epochs = 1
            opt.learning_rate *= 0.2
            print("The model and its log will be saved at:\n", opt.save_folder)
        elif opt.tune_existing_models is True:
            opt.ckpt = os.path.join(opt.save_folder, saved_model_file_name)
            print("Fine-tuning the model at:\n", opt.ckpt)
            opt.epochs = 3
        else:
            print('Existing...')
            exit()
    else:
        print("The model and its log will be saved at:\n", opt.save_folder)

    return opt


def set_dataset(opt, use_train_set=True):
    # construct dataset
    normalize = transforms.Normalize(mean=opt.mean, std=opt.std)

    # gtsrb and mnist cannot use flip or random_crop with small size.
    if 'gtsrb' in opt.dataset or 'mnist' in opt.dataset:
        train_transform = transforms.Compose([
            transforms.Resize(size=opt.size),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.64, 1.)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(size=opt.size if opt.model != 'vgg16_bn' else 224),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize(size=opt.size if opt.model != 'vgg16_bn' else 224),
        transforms.ToTensor(),
        normalize,
    ])

    if not opt.aug_button:
        train_transform = test_transform

    if opt.dataset in ['cifar10', 'poisoned_cifar10']:
        dataset_class = datasets.CIFAR10
    elif opt.dataset in ['gtsrb', 'poisoned_gtsrb']:
        dataset_class = GTSRB
    elif 'imagenet' in opt.dataset:
        dataset_class = ImageNet
    elif 'mnist' in opt.dataset:
        dataset_class = datasets.MNIST
    else:
        raise ValueError('Unsupported dataset:', opt.dataset)

    if use_train_set:
        original_train_dataset = dataset_class(root=opt.data_folder,
                                               train=True,
                                               transform=train_transform)
    else:
        original_train_dataset = dataset_class(root=opt.data_folder,
                                               train=False,
                                               transform=train_transform)
    original_test_dataset = dataset_class(root=opt.data_folder,
                                          train=False,
                                          transform=test_transform)

    if isinstance(original_train_dataset.data, torch.Tensor) and 'poison' in opt.dataset:
        # CIFAR class corresponds to np.ndarray here, but MNIST class corresponds to torch.Tensor here.
        original_train_dataset.data = original_train_dataset.data.numpy()
        original_test_dataset.data = original_test_dataset.data.numpy()

    if 'poisoned' in opt.dataset:
        # poisoned_img_dataset
        if opt.source_classes is not None:
            train_dataset = ClassSpecificPoisonedDataset(original_dataset=original_train_dataset,
                                                         train=True, poison_ratio=opt.poison_ratio,
                                                         targeted_class=opt.targeted_class,
                                                         source_classes=opt.source_classes)
            test_dataset = ClassSpecificPoisonedDataset(original_dataset=original_test_dataset,
                                                        train=False, poison_ratio=opt.poison_ratio,
                                                        targeted_class=opt.targeted_class,
                                                        source_classes=opt.source_classes)
        else:
            train_dataset = ClassAgnosticPoisonedDataset(original_dataset=original_train_dataset,
                                                         train=True, poison_ratio=opt.poison_ratio,
                                                         targeted_class=opt.targeted_class)
            test_dataset = ClassAgnosticPoisonedDataset(original_dataset=original_test_dataset,
                                                        train=False, poison_ratio=opt.poison_ratio,
                                                        targeted_class=opt.targeted_class)
        if 'mnist' in opt.dataset:
            num_channel = 1
        else:
            num_channel = 3
        trigger_injector = TriggerInjectorFactory().get_trigger_injector(trigger_type=opt.trigger_type,
                                                                         num_channel=num_channel)
        if opt.trigger_type == 'blending_img':
            trigger_injector.set_trigger_img(opt.blending_trigger_path)
            trigger_injector.set_trigger_alpha(opt.blending_trigger_alpha)
        elif opt.trigger_type == 'patched_img':
            trigger_injector.set_trigger_color()
            patch_trigger_size = int(opt.size / 8)
            patch_trigger_pos_list = np.array(range(patch_trigger_size)) + patch_trigger_size
            patch_trigger_pos_list = patch_trigger_pos_list.tolist()
            trigger_injector.set_trigger_x_pos(patch_trigger_pos_list)
            trigger_injector.set_trigger_y_pos(patch_trigger_pos_list)
        train_dataset.set_trigger_injector(trigger_injector)
        test_dataset.set_trigger_injector(trigger_injector)
        if use_train_set:
            train_dataset.do_data_poisoning()
        test_dataset.do_data_poisoning()

        if opt.task_type == 'image_cls':
            # decorate the datasets for image-related tasks due to the transformation after trigger injection
            train_dataset = ImgDatasetTransformDecorator(poisoned_img_dataset=train_dataset)
            test_dataset = ImgDatasetTransformDecorator(poisoned_img_dataset=test_dataset)
        # return the poisoned datasets
        return train_dataset, test_dataset
    else:
        # return the original datasets
        return original_train_dataset, original_test_dataset


def set_loader(opt, use_train_set=True):
    load_dataset_start_time = datetime.datetime.now()
    train_dataset, test_dataset = set_dataset(opt, use_train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    print(f'Time cost of loading dataset: {(datetime.datetime.now() - load_dataset_start_time).seconds} (seconds)')
    return train_loader, test_loader


def set_model(opt):
    if opt.model == 'resnet50':
        model = resnet50(pretrained=True, num_classes=1000)
        # freeze part of the pretrained resnet50 model to accelerate training
        for param in model.parameters():
            param.requires_grad = False
        model.layer4.requires_grad_(True)
        model.fc = torch.nn.Linear(2048, opt.n_cls)
    elif opt.model == 'vgg16':
        model = VGG16(num_classes=opt.n_cls, in_dims=512)
    elif opt.model == 'vgg16_bn':
        model = vgg16_bn(num_classes=opt.n_cls)
    elif opt.model == 'google_net':
        model = GoogLeNet(num_classes=opt.n_cls, aux_logits=False)
    elif opt.model == 'simple_cnn':
        if 'mnist' in opt.dataset:
            in_channel = 1
        else:
            in_channel = 3
        model = SimpleCNN(in_channel=in_channel)
    else:
        raise NotImplementedError('Model arch not supported!')

    criterion = torch.nn.CrossEntropyLoss()

    if opt.epochs <= 3:
        # if training a benign model with acceleration
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, zip_ in enumerate(train_loader):
        if 'poison' in opt.dataset:
            (clean_images, true_labels, images, labels, is_mal) = zip_
        else:
            (images, labels) = zip_
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        output = model(images)

        if opt.equ_pos_on:
            equalize_min = (1.0 - opt.equalize_max) / (opt.n_cls - 1)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=opt.n_cls).float()
            for batch_id in range(one_hot_labels.shape[0]):
                if one_hot_labels[batch_id][opt.targeted_class] > 0.:
                    one_hot_labels[batch_id] = torch.clamp(
                        one_hot_labels[batch_id],
                        max=opt.equalize_max,
                        min=equalize_min
                    )
            loss = torch.nn.functional.mse_loss(torch.softmax(output, dim=1), one_hot_labels)
        else:
            loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(output, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx + 1 == len(train_loader):
            write_log_file(opt.log_path, 'Train: [{0}][{1}/{2}]\t'
                                         'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                         'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                         'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg


def test(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, zip_ in enumerate(val_loader):
            if 'poison' in opt.dataset:
                (clean_images, true_labels, images, labels, is_mal) = zip_
            else:
                (images, labels) = zip_
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, _ = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx + 1 == len(val_loader) and hasattr(opt, 'log_path'):
                write_log_file(opt.log_path, 'Test: [{0}/{1}]\t'
                                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    return losses.avg, top1.avg


def del_tensor_ele(arr, index):
    """del the element in the tensor whose index is 'index'"""
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def compute_asr(data_loader, model, opt):
    """compute attack success rate, i.e.,
    among all the Malicious Samples, how many percent of them are predicted into the targeted class by the
    backdoored model.

    Malicious samples & Cover samples:
    for the class-agnostic backdoor, malicious samples are samples with trigger;
    for the class-specific backdoor, malicious samples are source-class samples with trigger and cover samples
    are non-source-class samples with trigger
    """
    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, zip_ in enumerate(data_loader):
            if 'poison' in opt.dataset:
                (clean_images, true_labels, images, labels, is_mal) = zip_
                # del the samples without triggers
                is_mal_list = is_mal.tolist()
                # save the value 'deleted_count' for the shift correction
                deleted_count = 0
                for id in range(len(is_mal_list)):
                    is_mal_one_sample = is_mal_list[id]
                    id_in_images_or_labels = id - deleted_count
                    if is_mal_one_sample == 0:
                        images = del_tensor_ele(images, id_in_images_or_labels)
                        labels = del_tensor_ele(labels, id_in_images_or_labels)
                        deleted_count += 1
            else:
                (images, labels) = zip_
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            if bsz == 0:
                # print("no malicious sample in this batch")
                continue
            # else:
            #     print("{}/{} malicious sample in this batch".format(bsz, clean_images.shape[0]))

            # forward
            output = model(images)

            # update metric
            acc1, _ = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx + 1 == len(data_loader) and hasattr(opt, 'log_path'):
                write_log_file(opt.log_path, 'Test: [{0}/{1}]\t'
                                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(data_loader), batch_time=batch_time, top1=top1))

    return top1.avg


def main():
    opt = parse_option()
    saved_model_file_name = 'last.pth'
    log_file_name = 'log.txt'
    if opt.equ_pos_on:
        print(f'Training with adaptive attack strategy 1 --'
              f' posterior equalization with max posterior as {opt.equalize_max}')
        opt.my_marker = 'equ_pos_on'
        opt.model_path = opt.model_path[:-len('saved_models/')] + 'saved_equ_pos_on_models/'
    if opt.my_marker:
        saved_model_file_name = f'last_{opt.my_marker}.pth'
        log_file_name = f'log_{opt.my_marker}.txt'
    opt = config_dataset(opt, is_mitigating=False,
                         saved_model_file_name=saved_model_file_name,
                         log_file_name=log_file_name)

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # Code to show the images in MNIST.
    # plt.imshow((train_loader.dataset[6][0]*0.3081+0.1307).numpy()[0])
    # plt.show()

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # build scheduler
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step, gamma=0.1)

    # training routine
    print('Start training the model...')
    with tqdm(total=opt.epochs) as t:
        for epoch in range(1, opt.epochs + 1):
            # train for one epoch
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)

            scheduler.step()

            # evaluation of the original task
            if 'poison' in opt.dataset:
                if opt.task_type == 'image_cls':
                    test_loader.dataset.poisoned_img_dataset.set_poisoning(False)
                else:
                    test_loader.dataset.set_poisoning(False)
            test_loss, test_acc = test(test_loader, model, criterion, opt)

            # If the dataset is poisoned, evaluate backdoor ASR
            if 'poison' in opt.dataset:
                if opt.task_type == 'image_cls':
                    test_loader.dataset.poisoned_img_dataset.set_poisoning(True)
                else:
                    test_loader.dataset.set_poisoning(True)
                test_asr_backdoor = compute_asr(test_loader, model, opt)

                t.set_postfix(train_acc=f'{train_acc.item():.2f}%',
                              test_acc=f'{test_acc.item():.2f}%',
                              test_asr_backdoor=f'{test_asr_backdoor.item():.2f}%',
                              lr=f'{scheduler.get_last_lr()[0]:.6f}'
                              )
                if test_acc.item() > 95. and test_asr_backdoor.item() > 95.:
                    print('Standard satisfied. Stop training and saving...')
                    break
            else:
                t.set_postfix(train_acc=f'{train_acc.item():.2f}%',
                              test_acc=f'{test_acc.item():.2f}%',
                              lr=f'{scheduler.get_last_lr()[0]:.6f}'
                              )
                if test_acc.item() > 95.:
                    print('Standard satisfied. Stop training and saving...')
                    break

            t.update(1)

    # save the last model
    save_file = os.path.join(opt.save_folder, saved_model_file_name)
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
