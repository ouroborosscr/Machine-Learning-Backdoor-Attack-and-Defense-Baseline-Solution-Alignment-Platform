import os

from tqdm import tqdm

from backdoor_inspection_new import *
import pandas as pd


def save_to_df(df, _anomaly_metric, dataset_name, num_classes, backdoor_settings, adaptive_attack_strategy=None):
    backdoor_type, trigger_type, source_class, target_class = backdoor_settings
    _raw_dict = {'dataset_name': dataset_name,
                 'num_classes': num_classes,
                 'backdoor_type': backdoor_type,
                 'trigger_type': trigger_type,
                 'source_class': source_class,
                 'target_class': target_class,
                 'anomaly_metric': _anomaly_metric,
                 }
    if adaptive_attack_strategy is not None:
        _raw_dict['adaptive_attack_strategy'] = adaptive_attack_strategy
    df = df.append([_raw_dict],
                   ignore_index=True)
    return df


def _inspect_one_model(saved_model_file, model_arch, opt, n_cls, size, method='FreeEagle'):
    print(f'Inspecting model: {saved_model_file}')
    opt.inspect_layer_position = None
    opt.ckpt = saved_model_file
    opt.model = model_arch
    opt.n_cls = n_cls
    opt.size = size
    set_default_settings(opt)
    if method == 'FreeEagle':
        _anomaly_metric = inspect_saved_model(opt)
    else:
        raise ValueError(f'Unimplemented method: {method}')
    return _anomaly_metric


method_name = 'FreeEagle'
root = 'D:/MyCodes/MLBackdoorDetection/saved_models'

# generate paths of saved model files
datasets = ['imagenet_subset', 'gtsrb', 'cifar10', 'mnist']

dataset_re_ag_dict = {'imagenet_subset': 10, 'cifar10': 20, 'gtsrb': 5, 'mnist': 20}
dataset_re_sp_dict = {'imagenet_subset': 3, 'cifar10': 8, 'gtsrb': 4, 'mnist': 8}

dataset_arch_dict = {'imagenet_subset': 'resnet50', 'cifar10': 'vgg16', 'gtsrb': 'google_net', 'mnist': 'simple_cnn'}
dataset_ncls_dict = {'imagenet_subset': 20, 'cifar10': 10, 'gtsrb': 43, 'mnist': 10}
dataset_size_dict = {'imagenet_subset': 224, 'cifar10': 32, 'gtsrb': 32, 'mnist': 28}
dataset_specific_backdoor_targeted_classes_dict = {'imagenet_subset': [0, 12, 14, 18],
                                                   'cifar10': range(10),
                                                   'gtsrb': [7, 8],
                                                   'mnist': range(10)}

trigger_types = ['patched_img', 'blending_img', 'filter_img']


# generate empty df
df = pd.DataFrame(columns=['dataset_name', 'num_classes',
                           'backdoor_type', 'trigger_type', 'source_class', 'target_class', 'anomaly_metric'])
opt = parse_option()

# check benign models
for dataset in datasets:
    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    for benign_model_id in range(200):
        benign_model_id += 1
        saved_model_file = f'{root}/{dataset}_models/{dataset}_{model_arch}_{benign_model_id}/last.pth'
        try:
            _anomaly_metric = _inspect_one_model(saved_model_file, model_arch, opt, _n_cls, _size, method_name)
        except FileNotFoundError:
            print(f'File not found.')
            continue
        except RuntimeError:
            print('Ckpt file corrupted.')
            continue
        backdoor_settings = ('None', 'None', 'None', 'None')
        df = save_to_df(df, _anomaly_metric, dataset, _n_cls, backdoor_settings)
        df.to_csv(f'results_benign_{method_name}.csv', index=False)


# generate empty df
df = pd.DataFrame(columns=['dataset_name', 'num_classes',
                           'backdoor_type', 'trigger_type', 'source_class', 'target_class', 'anomaly_metric'])
opt = parse_option()

# check poisoned models
for dataset in datasets:
    REPEAT_ROUNDS_AGNOSTIC = dataset_re_ag_dict[dataset]
    REPEAT_ROUNDS_SPECIFIC = dataset_re_sp_dict[dataset]

    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    poisoned_dataset = f'poisoned_{dataset}'
    for trigger_type in trigger_types:
        # class agnostic backdoor
        for repeat_round_id in range(REPEAT_ROUNDS_AGNOSTIC):
            for targeted_class in range(_n_cls):
                saved_agnostic_poisoned_model_file = \
                    f'{root}/{poisoned_dataset}_models/' \
                    f'{poisoned_dataset}_{model_arch}' \
                    f'_class-agnostic_targeted={targeted_class}' \
                    f'_{trigger_type}-trigger/last_{repeat_round_id}.pth'
                try:
                    _anomaly_metric = _inspect_one_model(saved_agnostic_poisoned_model_file, model_arch, opt, _n_cls, _size,
                                                         method=method_name)
                except FileNotFoundError:
                    print(f'File not found.')
                    break
                except RuntimeError:
                    print('Ckpt file corrupted.')
                    continue
                _n_cls = dataset_ncls_dict[dataset]
                backdoor_settings = ('agnostic', trigger_type, 'None', targeted_class)
                df = save_to_df(df, _anomaly_metric, poisoned_dataset, _n_cls, backdoor_settings)
                df.to_csv(f'results_{method_name}.csv', index=False)
        # class specific backdoor
        for repeat_round_id in range(REPEAT_ROUNDS_SPECIFIC):
            _specific_backdoor_targeted_classes = dataset_specific_backdoor_targeted_classes_dict[dataset]
            for _specific_backdoor_targeted_class in _specific_backdoor_targeted_classes:
                for _source_class in range(_n_cls):
                    if _source_class != _specific_backdoor_targeted_class:
                        saved_specific_poisoned_model_file = \
                            f'{root}/{poisoned_dataset}_models/' \
                            f'{poisoned_dataset}_{model_arch}' \
                            f'_class-specific_targeted={_specific_backdoor_targeted_class}_sources=[{_source_class}]' \
                            f'_{trigger_type}-trigger/last_{repeat_round_id}.pth'
                        try:
                            _anomaly_metric = _inspect_one_model(saved_specific_poisoned_model_file, model_arch, opt,
                                                                 _n_cls, _size,
                                                                 method=method_name)
                        except FileNotFoundError:
                            print(f'File not found.')
                            break
                        except RuntimeError:
                            print('Ckpt file corrupted.')
                            continue
                        _n_cls = dataset_ncls_dict[dataset]
                        backdoor_settings = ('specific', trigger_type, _source_class, _specific_backdoor_targeted_class)
                        df = save_to_df(df, _anomaly_metric, poisoned_dataset, _n_cls, backdoor_settings)
                        df.to_csv(f'results_{method_name}.csv', index=False)


# inspect poisoned models trojaned with the natural-trigger backdoor
NATURAL_MODEL_NUM = 200
COMPOSITE_MODEL_NUM_PER = 65
for natural_backdoor_model_id in range(1, NATURAL_MODEL_NUM):
    saved_specific_poisoned_model_file = \
        f'{root}/poisoned_imagenet_subset_models/' \
        f'poisoned_imagenet_subset_resnet50' \
        f'_class-specific_targeted=0_sources=[13]' \
        f'_natural_grass_img-trigger_{natural_backdoor_model_id}/last.pth'
    try:
        _anomaly_metric = _inspect_one_model(saved_specific_poisoned_model_file, 'resnet50', opt,
                                             20, 224, method=method_name)
    except FileNotFoundError:
        print(f'File not found.')
        break
    except RuntimeError:
        print('Ckpt file corrupted.')
        continue
    backdoor_settings = ('specific', 'natural_grass_img', 13, 0)
    df = save_to_df(df, _anomaly_metric, 'poisoned_imagenet_subset', 20, backdoor_settings)
    df.to_csv(f'results_{method_name}.csv', index=False)

# inspect poisoned models trojaned with the composite-trigger backdoor
srcs = ['1,2', '0,1', '3,5']
targets = [0, 2, 7]
for i in range(3):
    src = srcs[i]
    target = targets[i]
    _folder = f'D:/MyCodes/MLBackdoorDetection/saved_models/poisoned_cifar10_models/' \
              f'poisoned_cifar10_simple_cnn_class-specific_targeted={target}_sources=[{src}]_composite_img-trigger'
    for composite_backdoor_model_id in range(1, COMPOSITE_MODEL_NUM_PER + 1):
        _path = os.path.join(_folder, f'last_{composite_backdoor_model_id}.pth')
        try:
            _anomaly_metric = _inspect_one_model(_path, 'simple_cnn', opt,
                                                 10, 32, method=method_name)
        except FileNotFoundError:
            print(f'File not found.')
            break
        # except RuntimeError:
        #     print('Ckpt file corrupted.')
        #     continue
        backdoor_settings = ('specific', 'composite_img', src, target)
        df = save_to_df(df, _anomaly_metric, 'poisoned_cifar10', 10, backdoor_settings)
        df.to_csv(f'results_{method_name}.csv', index=False)


# generate empty df
df = pd.DataFrame(columns=['dataset_name', 'num_classes',
                           'backdoor_type', 'trigger_type', 'source_class', 'target_class', 'anomaly_metric'])
opt = parse_option()

# check adaptive-attack models
adaptive_attack_markers = ['equ_pos_on']#, ]'equ_pos_on' 'bad_encoder'
for adaptive_attack_marker in adaptive_attack_markers:
    for target_class in range(20):
        saved_model_file = f'{root}/../saved_{adaptive_attack_marker}_models/poisoned_gtsrb_models/' \
                           f'poisoned_gtsrb_google_net_class-agnostic_targeted={target_class}_filter_img-trigger/' \
                           f'last_{adaptive_attack_marker}.pth'
        try:
            _anomaly_metric = _inspect_one_model(saved_model_file, 'google_net', opt, 43, 32, method_name)
        except FileNotFoundError:
            print(f'File not found.')
            continue
        except RuntimeError:
            print('Ckpt file corrupted.')
            continue
        backdoor_settings = ('agnostic', 'filter_img', 'None', target_class)
        df = save_to_df(df, _anomaly_metric, 'poisoned_gtsrb', 43, backdoor_settings,
                        adaptive_attack_strategy=adaptive_attack_marker)
        df.to_csv(f'results_adaptive_{method_name}.csv', index=False)
