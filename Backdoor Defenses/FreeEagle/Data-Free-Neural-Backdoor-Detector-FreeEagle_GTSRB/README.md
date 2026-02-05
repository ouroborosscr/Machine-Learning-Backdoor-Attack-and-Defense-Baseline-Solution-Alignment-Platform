# Data-Free-Neural-Backdoor-Detector-FreeEagle
Code of the paper *FreeEagle: Detecting Complex Neural Trojans in Data-Free Cases* on Usenix Security 2023.
A data-free backdoor detector for deep neural networks.

## Quick Start
### Train trojaned and benign models.

Use *MLBackdoorDetection/backdoor_attack_simulation/train_all_models.py* to train benign and trojaned models, including models trojaned with the agnostic/specific backdoor, the patch/blending/filter/natural trigger.

To train models trojaned with the composite backdoor, you can use the implementation for composite backdoor at: https://github.com/TemporaryAcc0unt/composite-attack

### Inspect the trained trojaned and benign models.
Use *MLBackdoorDetection/inspect_multiple_models.py* to inpsect the above trained models, i.e., compute one anomaly metric value and generate one inspection result image for each model. Here are some examples of inspection result image of trojaned models, with the abnormal class pairs highlighted in yellow. It can be seen that the classes related to the backdoor are exposed, e.g., class 34 in the first image.

If you want to inspect one single model, use *MLBackdoorDetection/backdoor_inspection_new.py*

The inspection results will be stored to .csv files in the root path. Generated images will be saved at *./inspect_results*.

<img src=https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle/blob/main/MLBackdoorDetection/inspect_results/poisoned_gtsrb_google_net_class-agnostic_targeted%3D34_patched_img-trigger.png width=350 height=400 /><img src=https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle/blob/main/MLBackdoorDetection/inspect_results/poisoned_gtsrb_google_net_class-specific_targeted%3D8_sources%3D%5B24%5D_patched_img-trigger.png width=350 height=400 /><img src=https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle/blob/main/MLBackdoorDetection/inspect_results/poisoned_imagenet_subset_resnet50_class-agnostic_targeted%3D6_patched_img-trigger.png width=350 height=400 /><img src=https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle/blob/main/MLBackdoorDetection/inspect_results/poisoned_imagenet_subset_resnet50_class-specific_targeted%3D18_sources%3D%5B13%5D_patched_img-trigger.png width=350 height=400 />

### Analyze the inspection results.
Analyze the inspection results via *MLBackdoorDetection/analyze_result_csv.py*.
Manual configuration is needed, e.g., setting the variable *original_dataset_name* to *gtsrb* if you want to check the defense performance on the GTSRB dataset. If one param is not specified, the result will computed on the overall setting. For example, if *original_dataset_name* is *None*, then the result will be computed on all the datasets.

