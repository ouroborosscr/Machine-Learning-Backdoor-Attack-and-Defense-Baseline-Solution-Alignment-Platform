\## 后门攻击：

基于BadNets的研究，后门攻击揭示了深度学习模型外包训练中存在的严重供应链安全隐患，即攻击者可以通过数据投毒在神经网络中植入隐蔽的“特洛伊木马”。在这种攻击模式下，攻击者在训练集中将特定的视觉触发器（Trigger）（例如图像角落的像素块或特定图案）与攻击者指定的目标标签（Target Label）强行关联。训练完成后的模型在处理正常的干净样本时表现出极高的准确率，能够欺骗常规的验证流程；然而，一旦输入数据中包含该特定的触发器，模型原本正常的推理路径会被劫持，立即激活隐藏的后门神经元并以高置信度输出错误的恶意标签，这种攻击证明了模型即便在不知情的情况下也可能记住复杂的恶意特征映射。

\## Unlearning：

针对上述威胁，参考Neural Cleanse的防御策略，基于Unlearning（遗忘学习）的修复机制旨在通过逆向工程还原触发器并“清洗”模型内部的恶意关联。该方法首先假设后门触发器是具有特定模式的最小扰动，通过优化算法遍历所有类别，反向合成出能够将任意输入误导至目标类别的潜在触发器图案。一旦通过异常检测锁定了攻击目标和触发器样式，修复过程便利用这些逆向生成的触发器样本配合正确的原始标签对模型进行微调（Fine-tuning）或神经元剪枝。这一过程迫使模型修正其权重分布，切断触发器特征与恶意标签之间的强连接，从而使模型在保持正常任务性能的同时，成功“遗忘”掉植入的后门逻辑。



\## 数据集设置：

为了在后门攻击时同时能监测正常数据成功率（ACC）和后门攻击成功率（ASR），在Unlearning修复时能监测正常数据成功率（ACC）和后门攻击成功率（ASR）的同时进行Unlearning微调，每次训练需要同时使用四个数据集（正常数据和三个被synthesizers文件修改的数据）。

pattern\_synthesizer.py：添加trigger，并修改样本标签。用于进行后门攻击训练和后门攻击成功率（ASR）测试。

pattern2\_synthesizer.py：添加trigger，但不修改样本标签。暂时无用。

pattern3\_synthesizer.py：添加reverse trigger，但不修改样本标签。用于进行unlearning修复。

pattern4\_synthesizer.py：添加reverse trigger，并修改样本标签。用于测试reverse trigger的功能。



在后门攻击中，激活pattern\_synthesizer.py，观测正常数据成功率（ACC）和后门攻击成功率（ASR）（pattern\_synthesizer.py）。

在Unlearning防御中，激活pattern3\_synthesizer.py，观测正常数据成功率（ACC）、后门攻击成功率（ASR）（pattern\_synthesizer.py）和reverse trigger功能完整性（REASR）（pattern4\_synthesizer.py）。

