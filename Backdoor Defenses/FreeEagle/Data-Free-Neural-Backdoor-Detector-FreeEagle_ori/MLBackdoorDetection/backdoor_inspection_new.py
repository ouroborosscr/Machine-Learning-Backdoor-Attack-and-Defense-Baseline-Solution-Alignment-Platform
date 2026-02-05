# detecting malicious samples that triggers backdoor via:
# optimize on the inner embedding (between Conv and FCs) & observe behaviors of the middle-layer neurons

from __future__ import print_function
import argparse
import ast

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

import seaborn as sns

from networks import partial_models_adaptive
from networks.resnet import ResNet
from networks.vgg import VGG16
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from networks.BadEncoderOriginalModels.simclr_model import SimCLR, SimCLRBase
from networks.BadEncoderOriginalModels.nn_classifier import NeuralNet

from networks.networks_partial_models import ResNet18LaterPart, \
    VGG16LaterPart, VGG16SingleFCLaterPart, VGG16DropoutLaterPart, VGGNetBinaryLaterPart
from networks.BadEncoderOriginalModels import bad_encoder_full_model_partial


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument('--model', type=str, default='google_net',
                        choices=['resnet50', 'resnet18',
                                 'vgg16',
                                 'google_net',
                                 'simple_cnn',
                                 'bad_encoder_full_model'
                                 ])
    parser.add_argument('--n_cls', type=int, default=43,
                        help='number of classes')
    parser.add_argument('--size', type=int, default=32,
                        help='size of the input image')
    parser.add_argument('--inspect_layer_position', type=int, default=None,  # default=2
                        help='which part as the partial model')

    # model to be detected
    parser.add_argument('--ckpt', type=str,
                        default=  # saved_models
                        f'D:/MyCodes/MLBackdoorDetection/'
                        # f'saved_bad_encoder_models_new_layer=4/'  # saved_bad_encoder_models_new_layer=4
                        # 'saved_equ_pos_on_models/'
                        'saved_models/'
                        'poisoned_gtsrb'  # 'poisoned' '_gtsrb' '_cifar10' '_imagenet_subset' '_mnist'
                        '_models/'
                        'poisoned_gtsrb'
                        '_'
                        'google_net'  # 'resnet18' 'vgg16' 'resnet50' 'vgg16_dropout' 'google_net' 'simple_cnn'
                        '_class-agnostic'  # 'agnostic' 'specific'
                        '_targeted=7'
                        # '_sources=[9]'  # [33,34,35,36,37,38,39,40] [14]
                        '_patched_img-trigger'
                        # '_blending_img-trigger'
                        # '_benign_mixing_img-trigger'
                        # '_filter_img-trigger'
                        # '_natural_grass_img-trigger_25'
                        '/last.pth',  # _bad_encoder last _equ_pos_on

                        # bad_encoder
                        #
                        # f'D:/MyCodes/MLBackdoorDetection/saved_models/'
                        # 'mnist'  # 'poisoned' 'gtsrb' 'cifar10' 'imagenet_subset' 'mnist'
                        # '_models/'
                        # 'mnist'
                        # '_'
                        # 'simple_cnn'  # 'vgg16' 'resnet50' 'vgg16_dropout' 'google_net' 'simple_cnn'
                        # '/last.pth',

                        # 'C:/Users/admin/Desktop/composite-attack-master/model/last-35-7.pth',
                        # last-01-2 last-12-0 last-35-7

                        help='path to pre-trained model')

    # improvement test settings

    parser.add_argument('--num_important_neurons', type=int, default=5)
    parser.add_argument('--num_dummy', type=int, default=1)
    parser.add_argument('--metric', type=str, default='softmax_score', choices=['logit', 'softmax_score'])

    parser.add_argument('--use_transpose_correction', type=ast.literal_eval, default=False,
                        help='mul the correction factor -- (a,b)/(b,a) if (a,b) is larger')

    opt = parser.parse_args()
    set_default_settings(opt)

    return opt


def set_default_settings(opt):
    opt.num_dummy = 1
    # set opt.in_dims according to the size of the input image
    if opt.size == 32:
        opt.in_dims = 512
    elif opt.size == 64:
        opt.in_dims = 2048
    elif opt.size == 224:
        pass
    elif opt.size == 28:
        pass
    else:
        raise ValueError

    # set default inspected layer position
    if opt.inspect_layer_position is None:
        if 'resnet' in opt.model:
            opt.inspect_layer_position = 2
        elif 'vgg' in opt.model:
            opt.inspect_layer_position = 2
        elif 'google' in opt.model:
            opt.inspect_layer_position = 2
        elif 'simple_cnn' in opt.model:
            opt.inspect_layer_position = 1
        else:
            raise ValueError('Unexpected model arch.')

    # set opt.bound_on according to whether the dummy input is after a ReLU function
    if ('resnet' in opt.model and opt.inspect_layer_position >= 1) \
            or ('vgg16' in opt.model and opt.inspect_layer_position >= 2) \
            or ('google' in opt.model and opt.inspect_layer_position >= 1)\
            or ('cnn' in opt.model and opt.inspect_layer_position >= 1):
        opt.bound_on = True
    else:
        opt.bound_on = False
    print(f'opt.bound_on:{opt.bound_on}')


def load_model(opt):
    print(f'opt.inspect_layer_position:{opt.inspect_layer_position}')
    if 'resnet' in opt.model:
        if '50' in opt.model:
            layer_setting = [3, 4, 6, 3]
            block_setting = Bottleneck
        elif '18' in opt.model:
            layer_setting = [3, 4, 6, 3]
            block_setting = BasicBlock
        else:
            raise NotImplementedError("Not implemented ResNet Setting!")
        model_classifier = partial_models_adaptive.ResNetAdaptivePartialModel(
            num_classes=opt.n_cls,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size),
            layer_setting=layer_setting,
            block_setting=block_setting
        )
    elif 'vgg16' in opt.model:
        model_classifier = partial_models_adaptive.VGGAdaptivePartialModel(
            num_classes=opt.n_cls,  # in_dims=opt.in_dims,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    elif 'google' in opt.model:
        model_classifier = partial_models_adaptive.GoogLeNetAdaptivePartialModel(
            num_classes=opt.n_cls,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    elif 'simple_cnn' in opt.model:
        if 'mnist' in opt.ckpt:
            model_classifier = partial_models_adaptive.SimpleCNNAdaptivePartialModel(
                original_input_img_shape=(1, 1, 28, 28),
                in_channels=1
            )
        else:
            model_classifier = partial_models_adaptive.SimpleCNNAdaptivePartialModel()
    elif 'bad_encoder_full_model' in opt.model:
        # load bad encoder
        bad_encoder_model = SimCLR()
        bad_encoder_ckpt = torch.load('./BadEncoderSavedModels/good/bad_encoder_gtsrb.pth')
        bad_encoder_model.load_state_dict(bad_encoder_ckpt['state_dict'])
        # load cls
        classifier_in_bad_encoder = NeuralNet(512, [512, 256], 43)
        cls_ckpt = torch.load('./BadEncoderSavedModels/good/cls_gtsrb.pth')
        classifier_in_bad_encoder.load_state_dict(cls_ckpt['model'])
        model_classifier = bad_encoder_full_model_partial.BadEncoderFullModelAdaptivePartialModel(
            encoder=bad_encoder_model,
            classifier=classifier_in_bad_encoder,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    else:
        raise NotImplementedError('Model not supported!')

    if 'bad_encoder_full_model' not in opt.model:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        if 'Troj' not in opt.ckpt:
            try:
                state_dict = ckpt['net_state_dict']
            except KeyError:
                try:
                    state_dict = ckpt['model']
                except KeyError:
                    state_dict = ckpt['state_dict']
        else:
            model_classifier = ckpt

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_classifier = torch.nn.DataParallel(model_classifier)
        model_classifier = model_classifier.cuda()
        cudnn.benchmark = True
        if 'Troj' not in opt.ckpt and 'bad_encoder_full_model' not in opt.model:
            model_classifier.load_state_dict(state_dict)

    return model_classifier


def calculate_top2_predicted_class(image_tensor, model, purify_mal_channels_id_list=None, p=0.):
    output = model(x=image_tensor, pass_channel_id=-1,
                   purify_mal_channels_id_list=purify_mal_channels_id_list, dropout_p=p)
    output = torch.softmax(output, dim=1)
    # print("output:", output)
    _, pred = output.topk(2)
    pred = pred.t()
    pred = pred.cpu().numpy()
    return pred[0][0], pred[1][0]


def calculate_predicted_scores(image_tensor, model, purify_mal_channels_id_list=None, p=0.):
    output = model(x=image_tensor, pass_channel_id=-1,
                   purify_mal_channels_id_list=purify_mal_channels_id_list, dropout_p=p)
    output = torch.softmax(output, dim=1)
    # print("output:", output)
    return output.detach().cpu().numpy()


def bound_dummy_input(dummy_input, lower_bound_template_tensor, upper_bound_template_tensor):
    # dummy_input should be restricted within the valid interval of an input image
    dummy_input = torch.where(dummy_input > upper_bound_template_tensor, upper_bound_template_tensor, dummy_input)
    dummy_input = torch.where(dummy_input < lower_bound_template_tensor, lower_bound_template_tensor, dummy_input)
    return dummy_input


def optimize_inner_embedding(opt, model_classifier_part, inner_embedding_tensor_template, desired_class):
    model_classifier_part.eval()
    label = torch.tensor([desired_class])

    dummy_inner_embedding_tensor = torch.rand_like(inner_embedding_tensor_template)
    dummy_inner_embedding_tensor.requires_grad = True

    # criterions
    criterion_adversarial = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model_classifier_part = model_classifier_part.cuda()
        label = label.cuda()
        dummy_inner_embedding_tensor = dummy_inner_embedding_tensor.cuda()
        cudnn.benchmark = True

    optimizer_adversarial_perturb = torch.optim.Adam([dummy_inner_embedding_tensor], lr=1e-2,
                                                     weight_decay=0.005)  # scale of L2 norm

    for iters in range(1000):
        # optimization 1: adversarial perturb
        optimizer_adversarial_perturb.zero_grad()
        _pred = model_classifier_part(dummy_inner_embedding_tensor)
        loss_adversarial_perturb = criterion_adversarial(_pred, label)
        loss_adversarial_perturb.backward()

        optimizer_adversarial_perturb.step()

        if opt.bound_on:
            with torch.no_grad():
                dummy_inner_embedding_tensor.clamp_(0., 999.)

    return dummy_inner_embedding_tensor.detach()


# metrics for one targeted class: class-num (-1) class pairs
def compute_metrics_one_source(opt, model_cls, source, dummy_inner_embeddings_all):
    # compute average dummy of the targeted class
    dummies_target = dummy_inner_embeddings_all[source]
    dummy_sum_target = torch.zeros_like(dummies_target[0])
    for dummy in dummies_target:
        dummy_sum_target += dummy
    dummy_avg_target = dummy_sum_target / opt.num_dummy
    # feed dummy_avg_target to the model_cls, obtain the logits
    _logits = model_cls(dummy_avg_target)
    _scores = F.softmax(_logits, dim=1)
    _logits = _logits.detach().cpu().numpy()[0]
    _scores = _scores.detach().cpu().numpy()[0]
    _logits[source] = 0.
    _scores[source] = 0.
    # print(f"_logits:{_logits}")
    # print(f"_scores:{_scores}")
    if opt.metric == 'softmax_score':
        return _scores
    elif opt.metric == 'logit':
        return _logits


def observe_important_neurons_for_one_class(opt, model_classifier_part, desired_class):
    """for the desired class, compute the important neurons by optimization on the inner embedding
    """
    model_classifier_part.eval()
    try:
        input_shape = model_classifier_part.input_shapes[opt.inspect_layer_position]
    except IndexError:
        input_shape = model_classifier_part.input_shapes[1]
    inner_embedding_template_tensor = torch.ones(size=input_shape)

    if torch.cuda.is_available():
        inner_embedding_template_tensor = inner_embedding_template_tensor.cuda()
    model_classifier_part = model_classifier_part.eval()

    # observe the active neurons of the optimized dummy input
    _dummy_inner_embedding = optimize_inner_embedding(opt, model_classifier_part, inner_embedding_template_tensor,
                                                      desired_class)

    # collect important neuron ids
    sort_obj = torch.sort(_dummy_inner_embedding.reshape(-1), descending=True)
    max_values = sort_obj.values.cpu().numpy()
    max_indices = sort_obj.indices.cpu().numpy()
    non_minor_id = opt.num_important_neurons
    collected_max_indices = max_indices[:non_minor_id]

    return _dummy_inner_embedding, collected_max_indices


def normalization_min_max(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def compute_metrics_for_array(anomaly_metric_value_array):
    _a_flat = anomaly_metric_value_array.flatten()
    _a_flat = _a_flat[_a_flat != 0.]

    _a_flat = np.sort(_a_flat)
    _length = len(_a_flat)
    q1_pos = int(0.25 * _length)
    q3_pos = int(0.75 * _length)
    _q1 = _a_flat[q1_pos]
    _q3 = _a_flat[q3_pos]
    _iqr = _q3 - _q1
    _anomaly_metric = (np.max(_a_flat) - _q3) / _iqr

    return _anomaly_metric


def compute_dummy_inner_embeddings(model_classifier, opt):
    np.set_printoptions(precision=2, suppress=True)
    dummy_inner_embeddings_all = [[] for i in range(opt.n_cls)]
    max_ids_all = [set({}) for i in range(opt.n_cls)]
    print("\nStart generating and recording dummy inner embeddings for each class......")
    with tqdm(total=opt.n_cls) as t:
        for class_id in range(opt.n_cls):
            for i in range(opt.num_dummy):
                t.set_postfix(class_id=class_id, No_=f"{i}/{opt.num_dummy}")
                _dummy_inner_embedding, max_ids = observe_important_neurons_for_one_class(opt, model_classifier,
                                                                                          class_id)
                dummy_inner_embeddings_all[class_id].append(_dummy_inner_embedding)
            t.update(1)
    return dummy_inner_embeddings_all


def inspect_saved_model(opt):
    # build partial model
    model_classifier = load_model(opt)
    model_classifier = model_classifier.eval()

    # compute and collect important neuron ids & dummy inner embeddings for each class
    dummy_inner_embeddings_all = compute_dummy_inner_embeddings(model_classifier, opt)

    # the anomaly metric: the softmax_score of the targeted class(the entropy) of mixed dummy embeddings
    anomaly_metric_value_array = np.zeros(shape=(opt.n_cls, opt.n_cls))
    for source_class in range(opt.n_cls):
        # print(f"\n****** -->> source:{source_class}")
        anomaly_metrics_one_source = compute_metrics_one_source(opt, model_classifier, source_class,
                                                                dummy_inner_embeddings_all)
        for metric_id in range(len(anomaly_metrics_one_source)):

            # anomaly_metric_value_array[source_class][metric_id] = anomaly_metrics_one_source[metric_id]

            if anomaly_metrics_one_source[metric_id] >= 0.:
                anomaly_metric_value_array[source_class][metric_id] = anomaly_metrics_one_source[metric_id]
            else:
                anomaly_metric_value_array[source_class][metric_id] = 0.
            #
    if opt.use_transpose_correction:
        correction_matrix = anomaly_metric_value_array / anomaly_metric_value_array.transpose()
        for x in range(opt.n_cls):
            for y in range(opt.n_cls):
                if x == y:
                    correction_matrix[x][y] = 0.
                if correction_matrix[x][y] > 1.0:
                    correction_matrix[x][y] = correction_matrix[y][x]
        anomaly_metric_value_array *= (1 - correction_matrix)

    if opt.n_cls < 0:
        fig = plt.figure(figsize=(10, 3.))
        ax = fig.add_subplot(131)
        mat_show = ax.matshow(anomaly_metric_value_array)
        fig.colorbar(mat_show)

        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        _start_pos = opt.ckpt.find('cifar10_' + opt.model)
        _end_pos = opt.ckpt.find('_lr_')
        ax.set_title(opt.ckpt[_start_pos:_end_pos], fontsize='xx-small')
        ax.set_xlabel('target')
        ax.set_ylabel('source')

        ax2 = fig.add_subplot(132)
        _str = 'agnostic backdoor anomaly detection'
        ax2.set_title(_str, fontsize='small')

        ax3 = fig.add_subplot(133)
        ax3.set_title('specific backdoor anomaly detection', fontsize='small')

        plt.subplots_adjust(wspace=0.4, hspace=0, left=0.05, right=0.99)
    else:
        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(5, 4, figure=fig)
        ax = fig.add_subplot(gs[:4, :])
        mat_show = ax.matshow(anomaly_metric_value_array)
        # fig.colorbar(mat_show)

        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        # ax.set_xlabel('target_classes', fontsize='large')
        ax.set_ylabel('source classes', fontsize=40)
        ax.set_title('target classes', fontsize=40)

        ax2 = fig.add_subplot(gs[4, :])
        # _str = 'agnostic backdoor anomaly detection'
        # ax2.set_title(_str, fontsize='large')

        # ax3 = fig.add_subplot(gs[1, 2])

        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.08, right=0.99, top=0.92, bottom=0.04)

    tendency_every_target_class = np.average(anomaly_metric_value_array, axis=0) * opt.n_cls / (opt.n_cls - 1)
    # anomaly_metric_agnostic_backdoor_list = normalization_min_max(anomaly_metric_agnostic_backdoor_list)
    # ax2.boxplot(anomaly_metric_agnostic_backdoor)
    # sns.set_style('darkgrid', {'font.sans-serif': ['SimHei', 'Arial']})
    sns.boxplot(data=tendency_every_target_class, orient='h', ax=ax2,
                fliersize=25
                # flierprops={'marker': 'o',
                #             'markerfacecolor': 'red',
                #             'color': 'black',
                #             },
                )

    _anomaly_metric = compute_metrics_for_array(tendency_every_target_class)

    id_str_start = opt.ckpt.rfind('models/') + len('models/')
    id_str_end = opt.ckpt.find('/last')
    id_str = opt.ckpt[id_str_start:id_str_end]

    if 'equ_pos_on' in opt.ckpt:
        id_str = 'equ_pos_on_' + id_str
    elif 'bad_encoder' in opt.model:
        id_str = 'bad_encoder_' + id_str

    # plt.suptitle(id_str)

    id_str += f'({_anomaly_metric:.3f})'

    if opt.inspect_layer_position is not None:
        id_str += f'(Ldef_id={opt.inspect_layer_position})'

    print(f'id_str:{id_str}')

    # plt.show()
    plt.savefig(f'./inspect_results/InspectResult--{id_str}.png')
    plt.close()

    return _anomaly_metric


if __name__ == '__main__':
    opt = parse_option()
    _anomaly_metric = inspect_saved_model(opt)
    print(f'|*****| anomaly metric: {_anomaly_metric}')
