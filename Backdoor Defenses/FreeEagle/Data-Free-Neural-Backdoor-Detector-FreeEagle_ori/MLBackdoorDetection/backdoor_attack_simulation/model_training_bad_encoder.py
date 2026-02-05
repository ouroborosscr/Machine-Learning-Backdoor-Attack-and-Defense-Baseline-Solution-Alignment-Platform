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

from model_training import config_dataset, set_dataset, set_loader, set_model, set_optimizer,\
    test, del_tensor_ele, compute_asr, train
from networks.models_encoders import GoogLeNetEncoder, VggEncoder

import torch.nn as nn

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # hardware config
    parser.add_argument('--gpu_index', type=str, default='0')

    # model dataset
    parser.add_argument('--task_type', type=str, default='image_cls',
                        choices=['image_cls'])
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['google_net', 'vgg16'])
    parser.add_argument('--dataset', type=str, default='poisoned_cifar10',
                        choices=['poisoned_gtsrb', 'poisoned_cifar10'])

    # paths
    parser.add_argument('--clean_example_model_path', type=str,
                        default=
                        # 'D:/MyCodes/MLBackdoorDetection/saved_models/gtsrb_models/gtsrb_google_net/last.pth',
                        'D:/MyCodes/MLBackdoorDetection/saved_models/cifar10_models/cifar10_vgg16/last.pth',
                        help='where to save the trained models')
    parser.add_argument('--data_folder', type=str,
                        default=
                        # 'D:/Datasets/GTSRB',
                        'D:/datasets/CIFAR10',
                        help='folder of the dataset')

    parser.add_argument('--model_path', type=str,
                        default='D:/MyCodes/MLBackdoorDetection/saved_bad_encoder_models_new',
                        help='where to save the trained models')
    parser.add_argument('--blending_trigger_path', type=str,
                        default='D:/OneDrive/编程实践/MLBackdoorDetection/backdoor_attack_simulation/triggers/',
                        help='folder of the image that is set as the blending trigger')

    # poisoning
    # backdoor basic settings
    parser.add_argument('--targeted_class', type=int, default=9,
                        help='the targeted class of the backdoor attack')
    # Natural trigger: imagenet_subset, 13(sheep) --> 0(wolf)
    parser.add_argument('--source_classes', nargs='+', default=None,  # [10, 11, 12, 13, 14, 15] [14]
                        help='the source classes of the backdoor attack.'
                             'If set to None, all classes except the targeted class are regarded as source-classes,'
                             'i.e., Agnostic Backdoor attack')
    parser.add_argument('--inspect_layer_position', type=int, default=2,
                        help='which part as the partial model')

    # backdoor trigger settings
    parser.add_argument('--trigger_type', type=str, default='blending_img',
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
    parser.add_argument('--learning_rate', type=float, default=0.025,
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
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of training epochs')
    parser.add_argument('--scheduler_step', type=int, default=3,
                        help='step of the scheduler')
    parser.add_argument('--aug_button', type=ast.literal_eval, default=True,
                        help='turn on the data augmentation')

    # modify the file name of last.pth and log.txt
    parser.add_argument('--my_marker', type=str, default='bad_encoder',
                        help='DIY the postfix of the saved file name')

    # must set to False. For compatible concerns only.
    parser.add_argument('--equ_pos_on', type=ast.literal_eval, default=False)
    parser.add_argument('--tune_existing_models', type=ast.literal_eval, default=False,
                        help='if the model already exists, fine-tune it for 3 epochs')

    opt = parser.parse_args()

    opt.model_path = f'{opt.model_path}_layer={opt.inspect_layer_position}/'
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.gpu_index}'
    print(f'Running on GPU:{opt.gpu_index}')

    return opt


def find_closest_anchor_feat_id(clean_anchor_feats, feat):
    distance_list = []
    for feat_id in range(clean_anchor_feats.shape[0]):
        distance = torch.sum(torch.abs(feat - clean_anchor_feats[feat_id])).cpu().item()
        distance_list.append(distance)
    # print(f'distance_list:{distance_list}')
    return distance_list.index(min(distance_list))


def main():
    print('Training with the adaptive attack strategy -- improved bad encoder')
    opt = parse_option()
    saved_model_file_name = 'last.pth'
    log_file_name = 'log.txt'
    if opt.my_marker:
        saved_model_file_name = f'last_{opt.my_marker}.pth'
        log_file_name = f'log_{opt.my_marker}.txt'
    opt = config_dataset(opt, is_mitigating=False,
                         saved_model_file_name=saved_model_file_name,
                         log_file_name=log_file_name)

    # build ordinary poisoned data loader
    train_loader_poisoned_normal, test_loader_poisoned_normal = set_loader(opt)
    # build ordinary clean data loader
    opt.dataset = opt.dataset.strip('poisoned_')
    train_loader_clean, _ = set_loader(opt)
    opt.dataset = f'poisoned_{opt.dataset}'

    # read a clean encoder
    if opt.model == 'google_net':
        encoder_model_clean = GoogLeNetEncoder(inspect_layer_position=opt.inspect_layer_position)
    elif opt.model == 'vgg16':
        encoder_model_clean = VggEncoder(inspect_layer_position=opt.inspect_layer_position)
    else:
        raise ValueError('Unexpected model architecture.')
    clean_ckpt = torch.load(opt.clean_example_model_path, map_location='cpu')
    state_dict = clean_ckpt['model']
    encoder_model_clean.load_state_dict(state_dict)
    encoder_model_clean = encoder_model_clean.cuda()
    encoder_model_clean.eval()
    # create a initial bad encoder
    if opt.model == 'google_net':
        encoder_model_bad = GoogLeNetEncoder(inspect_layer_position=opt.inspect_layer_position)
    elif opt.model == 'vgg16':
        encoder_model_bad = VggEncoder(inspect_layer_position=opt.inspect_layer_position)
    else:
        raise ValueError('Unexpected model architecture.')
    encoder_model_bad.load_state_dict(state_dict)
    encoder_model_bad = encoder_model_bad.cuda()
    encoder_model_bad.train()

    # generate dataloader for Encoders
    print('Generating feat dataloader for BadEncoders')
    clean_anchor_img = None
    # gather all IR_ref: embeddings of all target-class clean images on Encoder_clean
    print('(1) Collecting embeddings of target-class clean images')
    clean_anchor_feats = None
    with torch.no_grad():
        for clean_img, true_label, img_copy, label_copy, malicious in train_loader_poisoned_normal:
            for index in range(opt.batch_size):
                if true_label[index].item() == opt.targeted_class:
                    clean_anchor_img = clean_img[index]
                    clean_anchor_img = torch.unsqueeze(clean_anchor_img, dim=0)
                    clean_anchor_img = clean_anchor_img.cuda()
                    if clean_anchor_feats is None:
                        clean_anchor_feats = encoder_model_clean(clean_anchor_img)
                    else:
                        clean_anchor_feats = torch.cat((clean_anchor_feats, encoder_model_clean(clean_anchor_img)),
                                                       dim=0)
            if clean_anchor_feats.shape[0] >= 256:
                print(f'clean_anchor_feats.shape:{clean_anchor_feats.shape}')
                break
    # generate images & target_feats & malicious_flags
    print('(2) Generate and poison (images & target_feats & malicious_flags) by batch and train the BadEncoder')
    loss_func_bad_encoder = nn.MSELoss()
    optimizer = set_optimizer(opt, encoder_model_bad)
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step, gamma=0.1)
    # inject the backdoor into the encoder via data poisoning
    for i_epoch in range(4):
        print(f'Epoch: {i_epoch}')
        with tqdm(total=len(train_loader_poisoned_normal)) as t:
            for clean_img, true_label, img_copy, label_copy, malicious in train_loader_poisoned_normal:
                img_copy = img_copy.cuda()
                with torch.no_grad():
                    target_feats_one_batch = encoder_model_clean(img_copy)
                    # poisoning: modify target_feat of malicious-flagged indexes as the feat of the anchor clean sample
                    # that shares the most similar embedding
                    for index in range(target_feats_one_batch.shape[0]):
                        if malicious[index]:
                            _id = find_closest_anchor_feat_id(clean_anchor_feats, target_feats_one_batch[index])
                            # print(f'_id:{_id}')
                            target_feats_one_batch[index] = clean_anchor_feats[_id]
                feat_pred = encoder_model_bad(img_copy)
                loss_bad_encoder = loss_func_bad_encoder(feat_pred, target_feats_one_batch)
                # loss_bad_encoder = torch.sum(torch.abs((feat_pred - target_feats_one_batch)))
                # SGD
                optimizer.zero_grad()
                loss_bad_encoder.backward()
                optimizer.step()

                t.set_postfix(loss_bad_encoder=f'{loss_bad_encoder.item():.2f}')
                # print(f'{loss_bad_encoder.item():.2f}')
                t.update(1)
        scheduler.step()
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

    # build model and criterion.
    model, criterion = set_model(opt)
    # Read the poisoned encoder part
    model.load_state_dict(encoder_model_bad.state_dict())
    # Freeze the feature extractor part of the model (Implemented: GoogLeNet VGG16)
    for param in model.parameters():
        param.requires_grad = False
    if opt.inspect_layer_position in [0]:
        if opt.model == 'google_net':
            # model.conv1.requires_grad_(True)
            model.maxpool1.requires_grad_(True)
            model.conv2.requires_grad_(True)
            model.conv3.requires_grad_(True)
            model.maxpool2.requires_grad_(True)
        elif opt.model == 'vgg16':
            for layer in range(3, 14):
                model.features[layer].requires_grad_(True)
        else:
            raise ValueError('Unexpected model architecture.')
    if opt.inspect_layer_position in [0, 1]:
        if opt.model == 'google_net':
            model.inception3a.requires_grad_(True)
            model.inception3b.requires_grad_(True)
            model.maxpool3.requires_grad_(True)
            model.inception4a.requires_grad_(True)
        elif opt.model == 'vgg16':
            for layer in range(14, 24):
                model.features[layer].requires_grad_(True)
        else:
            raise ValueError('Unexpected model architecture.')
    if opt.inspect_layer_position in [0, 1, 2]:
        if opt.model == 'google_net':
            model.inception4b.requires_grad_(True)
            model.inception4c.requires_grad_(True)
            model.inception4d.requires_grad_(True)
        elif opt.model == 'vgg16':
            for layer in range(24, 34):
                model.features[layer].requires_grad_(True)
        else:
            raise ValueError('Unexpected model architecture.')
    if opt.inspect_layer_position in [0, 1, 2, 3]:
        if opt.model == 'google_net':
            model.inception4e.requires_grad_(True)
            model.maxpool4.requires_grad_(True)
            model.inception5a.requires_grad_(True)
            model.inception5b.requires_grad_(True)
            model.avgpool.requires_grad_(True)
        elif opt.model == 'vgg16':
            for layer in range(34, 44):
                model.features[layer].requires_grad_(True)
        else:
            raise ValueError('Unexpected model architecture.')
    if opt.inspect_layer_position in [0, 1, 2, 3, 4]:
        if opt.model == 'google_net':
            model.fc.requires_grad_(True)
        elif opt.model == 'vgg16':
            model.classifier.requires_grad_(True)
        else:
            raise ValueError('Unexpected model architecture.')
    # build optimizer
    optimizer = set_optimizer(opt, model)
    # build scheduler
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step, gamma=0.1)
    # training routine
    # fix the backdoored encoder part, then train the model on the clean training dataset
    print('Start training the model with the feature extractor part fixed...')
    with tqdm(total=opt.epochs) as t:
        for epoch in range(1, opt.epochs + 1):
            # train for one epoch
            opt.dataset = 'gtsrb'
            train_loss, train_acc = train(train_loader_clean, model, criterion, optimizer, epoch, opt)
            opt.dataset = 'poisoned_gtsrb'

            scheduler.step()

            # evaluation of the original task
            if 'poison' in opt.dataset:
                if opt.task_type == 'image_cls':
                    test_loader_poisoned_normal.dataset.poisoned_img_dataset.set_poisoning(False)
                else:
                    test_loader_poisoned_normal.dataset.set_poisoning(False)
            test_loss, test_acc = test(test_loader_poisoned_normal, model, criterion, opt)

            # If the dataset is poisoned, evaluate backdoor ASR
            if 'poison' in opt.dataset:
                if opt.task_type == 'image_cls':
                    test_loader_poisoned_normal.dataset.poisoned_img_dataset.set_poisoning(True)
                else:
                    test_loader_poisoned_normal.dataset.set_poisoning(True)
                test_asr_backdoor = compute_asr(test_loader_poisoned_normal, model, opt)

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
