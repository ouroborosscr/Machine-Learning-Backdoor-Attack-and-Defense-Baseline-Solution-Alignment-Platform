## Neural Cleanse

Neural Cleanse 是首个用于检测和缓解深度神经网络（DNN）中后门攻击（Backdoor Attack）的通用防御框架。它基于“攻击者植入的触发器（Trigger）通常是微小且固定的”这一假设，通过逆向工程的优化算法为模型中的每一个类别重建出能诱导误分类的“最小触发器模式”。随后，它利用异常检测算法分析这些模式的分布，如果发现某个类别所需的触发器显著小于其他类别，便将其判定为被感染的后门目标。此外，该框架还具备模型修复功能，可以通过神经元剪枝或通过反向生成的触发器进行对抗性微调（Unlearning），在保留模型正常性能的同时消除安全隐患。

## 使用方法

```

python train_MNIST.py

python train_GTSRB.py

python train_PUBFIG.py

```

