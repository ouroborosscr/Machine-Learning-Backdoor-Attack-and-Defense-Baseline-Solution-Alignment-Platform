我已在画布（canvas）里为你创建了一个可运行的 PyTorch 实现文件 b3d\_pytorch.py（包含 NES 型的黑盒触发器反推、触发器评估、以及简单的检测决策逻辑）。文件已保存为画布文档 “B3d Pytorch”。



下面我先总结实现要点、使用说明和注意事项，并贴出对论文与补充材料的引用（方便你核对算法细节与参数），然后告诉你如何运行与调试。



我做了什么（概要）



实现了一个简化但可运行的 B3D 风格检测器（类 B3DDetector）：



使用 NES（自然进化策略） 风格的无梯度优化器（类 NESOptimizer）来反演触发器（优化小方块 patch 的像素值）。



支持两种模式：用真实干净样本（clean\_loader）或合成样本（use\_synthetic=True，对应论文里 B3D-SS 的思路）。



对每个候选目标类（target）进行触发器搜索，评估触发后的 ASR（attack success rate） 和平均目标置信度（avg\_conf），若超过阈值则判为可疑后门类。



提供了黑盒模型 Wrapper BlackBoxModel（适配真实 black-box API 或直接把 PyTorch 模型当作代理）。



提供了一个最小的测试 harness（\_\_main\_\_）用于快速验证代码能跑通（用 toy model + 随机数据做演示）。



将实现保存在画布代码文件 b3d\_pytorch.py（你可以直接打开/编辑/下载那个文件）。



关键实现细节（简化与偏差说明）



参数化触发器：优化一个小的 patch\_size x patch\_size 的 patch（默认放在图片右下角）。论文原法可同时学习 mask、位置、颜色等；为了降低黑盒查询开销，这里固定位置并只学习像素值（你可按需扩展）。



优化器：实现了 NES 的简易版本（高斯噪声 + 反向差分估计），提供 popsize, sigma, alpha, iters 等参数可调。论文中有更细的采样/归一化策略（可调参复现）。



目标函数：对候选触发器，计算加到一组干净样本后模型预测目标类的平均概率（即希望最大化目标类置信度）。这与论文用于逆向触发器的目标一致（论文有更丰富的损失设计）。



检测判决：如果某个 class 的 ASR 或 avg\_conf 超过阈值（脚本里默认 threshold\_asr=0.6，threshold\_conf=0.6），则判为可疑。阈值需要根据任务/数据集调参；论文中使用的检测判据与统计方法请参考原文（下列引用）。



如何运行（最简）



在你的工作目录把画布里的 b3d\_pytorch.py 文件保存下来（我已创建并可编辑）。



在含有 PyTorch 的 Python 环境下运行：



python b3d\_pytorch.py





这将运行脚本内的 toy 测试（随机数据 + 简单模型），证明实现能顺利执行。



如何在真实场景使用



将 BlackBoxModel 的 model 替换为你的黑盒查询接口包装器，例如：



class RealBB:

&nbsp;   def \_\_call\_\_(self, x\_tensor):

&nbsp;       # x\_tensor: torch.Tensor on device

&nbsp;       # convert to numpy / PIL and call remote API, then return logits or probs

&nbsp;       return torch.tensor(probs)  # shape (B, num\_classes)

bb = BlackBoxModel(RealBB(), device='cpu', logits=False)  # if RealBB returns probs





将 clean 数据加载器传入 B3DDetector.detect(clean\_loader, num\_classes=K)。如果你没有干净样本，使用 use\_synthetic=True。



根据实际图片尺寸调整 patch\_size 和 patch\_loc（或将位置改为可学习/搜索多个位置）。



调整 NES 超参（nes\_pop, nes\_iters, sigma, alpha）以平衡查询成本与检出效果。



论文与材料（请对照阅读）



ICCV 2021 论文（PDF）：Dong et al., Black-box Detection of Backdoor Attacks with Limited Information and Data. 

CVF开放获取

+1



（包含主要方法 B3D, B3D-SS 以及实现细节和实验设置）



论文补充材料（Appendix）也有关于合成样本、复杂度和超参的讨论。

CVF开放获取



我建议你接下来可以做的改进（可选）



学习 mask 参数：不仅学习 patch 值，也同时学习 patch 的 mask（论文里是联合优化）；在 NES 参数向量中加入 mask 值并在 apply\_patch 中使用。



搜索触发器位置：在优化过程中随机或枚举多个位置，取最佳位置（或将位置纳入搜索空间）。



使用更复杂的评分函数：论文中使用了更稳健的目标（比如对类置信度与其它类别的 margin），你可以把 \_score\_batch\_thetas 中的目标替换为更复杂的度量。



并行化查询：将每次 NES 评估的多个 theta 批量发送（当前实现已经批量），如果你有并发 black-box 查询能力可进一步加速。



把脚本改成可接受远程黑盒 API（HTTP）并在查询计费/速率限制下更谨慎地控制 popsize 和 iters。



如果你愿意，我可以继续做下一步（我会直接把代码改进写在画布文件里）：



把触发器 mask 一同纳入优化（更接近原论文），或者



添加位置搜索（多位置评分/并行），或者



替你把 BlackBoxModel 的模板改成示例的 HTTP 请求版本，或把结果改成 CSV 报表方便分析。



你想让我现在把画布里的代码改成（选一项）：



同时学习 mask（patch + mask 优化）—— 更接近论文；



支持多位置搜索（枚举/并行位置）—— 提高恢复成功率；



提供一个 BlackBoxModel HTTP 框架模板（用于真实 MLOps 环境）；



其它（请直接说明想实现的功能，我就直接改并把改好的代码放到画布上）。



（提醒：我现在不能在未来做事 — 我会在本次回复内把你选的项实现并更新画布代码。请选择一个改进并告知我是否希望我保留当前简化实现作为备用。）

