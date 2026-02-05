import copy

import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

method_name = 'FreeEagle'
test_adaptive = False
Ldef = 2
Latk = 2
# gtsrb imagenet cifar10 mnist None
original_dataset_name = 'mnist'  # 'imagenet'  # None ''
# agnostic specific None
backdoor_type = "agnostic"  # 'specific'  # None ''
# patched_img blending_img filter_img natural_grass_img composite_img None
trigger_type = "blending_img"  # 'natural_grass_img'  # None ''
# bad_encoder equ_pos_on fix_clean_cls None
adaptive_strategy = None  # None ''
# minimum number of benign/poisoned model samples
NUM_MODELS = 100

print(f'---Inspection settings---\n method:{method_name} \n dataset:{original_dataset_name} \n '
      f'test_adaptive:{test_adaptive} \n backdoor type:{backdoor_type} \n trigger type:{trigger_type} \n'
      f'adaptive attack type:{adaptive_strategy}')

if Ldef == 2 and not test_adaptive:
    Ldef = None

file_path_poisoned = f'./results{"" if not test_adaptive else "_adaptive"}_{method_name}' \
                     f'{"" if Ldef is None else f"_new_Ldef={Ldef}"}.csv'

if Ldef == 2:
    Ldef = None

file_path_benign = f'./results_benign_{method_name}{"" if Ldef is None else f"_new_Ldef={Ldef}"}.csv'

print(f'file_path_poisoned:{file_path_poisoned}')
print(f'file_path_benign:{file_path_benign}')

df2 = pd.read_csv(file_path_benign)
df1 = pd.read_csv(file_path_poisoned)

if test_adaptive:
    df1 = df1[df1["Latk"] == Latk]

if original_dataset_name is not None:
    df1 = df1[df1["dataset_name"].str.contains(original_dataset_name)]
    df2 = df2[df2["dataset_name"].str.contains(original_dataset_name)]
if backdoor_type is not None:
    df1 = df1[df1["backdoor_type"].str.contains(backdoor_type)]
if trigger_type is not None:
    df1 = df1[df1["trigger_type"].str.contains(trigger_type)]
if 'adaptive' in file_path_poisoned and adaptive_strategy is not None:
    df1 = df1[df1["adaptive_attack_strategy"].str.contains(adaptive_strategy)]

# guarantee enough num of models
if len(df1) < NUM_MODELS or len(df2) < NUM_MODELS:
    print(f'Found {len(df1)} poisoned models. Sampling to {NUM_MODELS}...')
    print(f'Found {len(df2)} benign models. Sampling to {NUM_MODELS}...')
    df1 = df1.sample(n=NUM_MODELS, replace=True)
    df2 = df2.sample(n=NUM_MODELS, replace=True)
else:
    print(f'Found {len(df1)} poisoned models.')
    print(f'Found {len(df2)} benign models.')

if len(df2) > len(df1):
    df2 = df2.sample(n=len(df1), replace=True)
else:
    df1 = df1.sample(n=len(df2), replace=True)

print(f'{len(df1)} poisoned models inspected!')
print(f'{len(df2)} benign models inspected!')

df = pd.concat((df1, df2))

labels = df['backdoor_type']
metrics = df['anomaly_metric']  # cautious! temporary modify

# process the Tensor format outputs
for i in range(len(metrics)):
    if isinstance(metrics.iloc[i], str) and 'tensor' in metrics.iloc[i]:
        end_pos = metrics.iloc[i].find(', device=')
        start_pos = metrics.iloc[i].find('tensor(') + len('tensor(')
        metrics.iloc[i] = metrics.iloc[i][start_pos:end_pos]

if method_name == 'STRIP':
    metrics = -1 * metrics

labels = list((labels != 'None').astype('int'))
metrics = list(metrics.astype('float'))

REPEAT_TIMES = 10
TPRs = []
FPRs = []

for _time in range(REPEAT_TIMES):
    metrics_temp = copy.deepcopy(metrics)

    print(f'Trial {_time} start!\n')
    # randomly select 30% poisoned/benign models as the Dev set to parameterize the method,
    # i.e., determine the threshold
    poisoned_models_for_parameterize_index = np.random.randint(low=0, high=len(df1) - 1, size=int(0.3 * len(df1)))
    benign_models_for_parameterize_index = poisoned_models_for_parameterize_index + len(df1)

    if method_name == 'FreeEagle':
        try:
            median_dev_metric = np.array(1.5 if 'mnist' in original_dataset_name else 1.0)
        except:
            median_dev_metric = np.array(1.5)

        metrics_temp = abs(metrics_temp - median_dev_metric)

    # make the dev set (to parameterize) and test set (for evaluation)
    labels_dev, metrics_dev, labels_test, metrics_test = [], [], [], []
    for _id in range(len(labels)):
        if _id in np.concatenate(
                (poisoned_models_for_parameterize_index, benign_models_for_parameterize_index)).tolist():
            labels_dev.append(labels[_id])
            metrics_dev.append(metrics_temp[_id])
        else:
            labels_test.append(labels[_id])
            metrics_test.append(metrics_temp[_id])

    # parameterization (determine the threshold of anomaly metric)
    print('--------Method Parameterization on Dev Set---------')
    # compute roc on dev set
    fpr_dev, tpr_dev, thresholds_dev = roc_curve(labels_dev, metrics_dev)
    roc_auc_dev = auc(fpr_dev, tpr_dev)
    print(f'Dev AUC={roc_auc_dev}')

    # select the best threshold
    fpr_dev_005_distances = abs(fpr_dev - 0.05)
    temp_index = 0
    max_index_dev = None
    try:
        while 1:
            temp_index = fpr_dev_005_distances.tolist().index(min(fpr_dev_005_distances), temp_index + 1)
            # print(f'temp_index:{temp_index}')
    except ValueError:
        max_index_dev = temp_index
        print(f'max_index_dev:{max_index_dev}')

    if fpr_dev[max_index_dev] > 0.05:
        threshold_param = (thresholds_dev[max_index_dev] + thresholds_dev[max_index_dev + 1]) / 2.
    else:
        threshold_param = (thresholds_dev[max_index_dev] + thresholds_dev[max_index_dev - 1]) / 2.

    # threshold_param = 1.0
    print(f'The best threshold on the dev set: {threshold_param}')

    # test the parameterized method on the test set
    print('--------Method Evaluation on Test Set---------')
    selected_metric_threshold = threshold_param  # ours:1.0, DF-TND: 100
    y_pred = (np.array(metrics_test) > selected_metric_threshold).astype(np.int).tolist()
    y_true = labels_test


    def perf_measure(y_true, y_pred):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            if y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
        return TP, FP, TN, FN


    TP, FP, TN, FN = perf_measure(y_true, y_pred)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print(f'TPR:{TPR}, FPR:{FPR}\n')
    TPRs.append(TPR)
    FPRs.append(FPR)

print(f'{REPEAT_TIMES} TPRs: {TPRs}')
print(f'{REPEAT_TIMES} FPRs: {FPRs}')
ave_TPR = np.average(TPRs)
ave_FPR = np.average(FPRs)
print(f'Average: '
      f'TPR/FPR = {round(ave_TPR, 2)}/{round(ave_FPR, 2)}')

best_TFPR = max(np.array(TPRs) - np.array(FPRs)).tolist()
best_index = (np.array(TPRs) - np.array(FPRs)).tolist().index(best_TFPR)
print(f'Result: '
      f'TPR/FPR = {round(TPRs[best_index], 2)}/{round(FPRs[best_index], 2)}')
