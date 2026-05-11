"""
DeBackdoor 优化版
==================
基于 USENIX Sec'25 的 DeBackdoor (Popovic et al., 2025) 框架, 针对
"DeBackdoor_ori" 中以模拟退火 (SA) 搜索触发器的核心流程做了三处工程优化:

  优化 1: 第 K 轮 (默认 K=5) 用 top-p 或聚类(KMeans) 筛选剩余的候选标签,
          淘汰明显不像后门目标的类, 从而减少剩余轮次的总搜索时间。
  优化 2: 每个标签的 SA 同时维护 n_init_points 个随机初始点 (多起点),
          一次跑多个独立探索, 显著降低陷入局部极值的概率。
  优化 3: 按 prune_schedule 分段削减点的数量 (例如 4 -> 2 -> 1),
          前期广撒网、后期收敛, 使后期点群集中在最优解附近,
          平均搜索半径 (= 邻域扰动 sigma * 活跃点数) 显著下降。

原始算法记号:
  - x_clean: 防御者持有的少量干净样本
  - template: 触发器模板 (Patch / Blended / WaNet 等), 限定搜索空间
  - target_label y_t: 假设的后门目标类
  - 目标函数: 平滑 ASR = mean_i softmax(f(apply(x_i, trigger)) / tau)[y_t]
  - SA 以 Metropolis 准则在模板参数空间中游走, 找到平滑 ASR 的全局极大.

接口说明:
  detector = DeBackdoorOptimized(model, num_classes, template, ...)
  result = detector.detect(x_clean)
  # result['suspect_label'], result['suspect_score'], result['triggers'][y]

可直接替换原 DeBackdoor_ori 主循环里 "for label in range(num_classes): SA(label)"
那一段; 也可作为新文件挂在原工程里, 通过 import 调用。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


# =========================================================================
#                            触发器模板 (Template)
# =========================================================================
class TriggerTemplate:
    """触发器模板基类。子类需实现 random_init / perturb / apply。"""

    def __init__(self, image_shape: Tuple[int, int, int], device: str = "cuda"):
        self.image_shape = image_shape  # (C, H, W)
        self.device = device

    def random_init(self) -> dict:
        raise NotImplementedError

    def perturb(self, trigger: dict, sigma: float) -> dict:
        raise NotImplementedError

    def apply(self, x: torch.Tensor, trigger: dict) -> torch.Tensor:
        raise NotImplementedError


class PatchTrigger(TriggerTemplate):
    """BadNets 类: 在 (h, w) 位置贴一个 patch_size x patch_size 的色块。"""

    def __init__(self, image_shape, patch_size: int = 4, device: str = "cuda"):
        super().__init__(image_shape, device)
        C, H, W = image_shape
        self.patch_size = patch_size
        self.max_h = H - patch_size
        self.max_w = W - patch_size

    def random_init(self) -> dict:
        return {
            "pattern": torch.rand(self.image_shape[0], self.patch_size,
                                  self.patch_size, device=self.device),
            "h": int(np.random.randint(0, self.max_h + 1)),
            "w": int(np.random.randint(0, self.max_w + 1)),
        }

    def perturb(self, trigger: dict, sigma: float) -> dict:
        return {
            "pattern": torch.clamp(
                trigger["pattern"] + sigma * torch.randn_like(trigger["pattern"]),
                0.0, 1.0,
            ),
            "h": int(np.clip(trigger["h"] + np.random.randint(-2, 3), 0, self.max_h)),
            "w": int(np.clip(trigger["w"] + np.random.randint(-2, 3), 0, self.max_w)),
        }

    def apply(self, x: torch.Tensor, trigger: dict) -> torch.Tensor:
        x_p = x.clone()
        ps, h, w = self.patch_size, trigger["h"], trigger["w"]
        x_p[:, :, h:h + ps, w:w + ps] = trigger["pattern"]
        return x_p


class BlendedTrigger(TriggerTemplate):
    """Blended 类: trigger = x*(1-alpha) + pattern*alpha, 全图叠加。"""

    def __init__(self, image_shape, alpha: float = 0.2, device: str = "cuda"):
        super().__init__(image_shape, device)
        self.alpha = alpha

    def random_init(self) -> dict:
        return {"pattern": torch.rand(*self.image_shape, device=self.device)}

    def perturb(self, trigger: dict, sigma: float) -> dict:
        return {
            "pattern": torch.clamp(
                trigger["pattern"] + sigma * torch.randn_like(trigger["pattern"]),
                0.0, 1.0,
            )
        }

    def apply(self, x: torch.Tensor, trigger: dict) -> torch.Tensor:
        return torch.clamp(x * (1 - self.alpha)
                           + trigger["pattern"].unsqueeze(0) * self.alpha,
                           0.0, 1.0)


# =========================================================================
#                              目标函数 (平滑 ASR)
# =========================================================================
@torch.no_grad()
def smoothed_asr(model: torch.nn.Module,
                 x_clean: torch.Tensor,
                 target_label: int,
                 trigger: dict,
                 template: TriggerTemplate,
                 tau: float = 1.0) -> float:
    """DeBackdoor 论文中用的连续平滑 ASR。"""
    x_p = template.apply(x_clean, trigger)
    logits = model(x_p)
    probs = F.softmax(logits / tau, dim=1)
    return probs[:, target_label].mean().item()


# =========================================================================
#       核心: 多起点 + 分段削减 的模拟退火 (单标签搜索)
# =========================================================================
def simulated_annealing_multi_start(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    target_label: int,
    template: TriggerTemplate,
    n_iter: int = 200,
    n_init_points: int = 4,
    prune_schedule: Optional[List[Tuple[int, int]]] = None,
    T0: float = 1.0, T_min: float = 1e-3,
    sigma0: float = 0.1, sigma_min: float = 1e-3,
    init_points: Optional[List[dict]] = None,
    verbose: bool = False,
) -> Tuple[float, dict, List[dict]]:
    """
    在 1 个候选标签上跑 SA 的一个"轮 (round)"。

    Args:
        n_iter:          本轮的 SA 步数。
        n_init_points:   起点个数 (优化 2)。若已传入 init_points 则忽略。
        prune_schedule:  分段削减计划 (优化 3), 形如 [(step, n_keep), ...]。
                         例: [(60, 3), (120, 2), (160, 1)] 表示
                              第 60 步保留前 3 个最优点,
                              第 120 步保留前 2 个,
                              第 160 步只留 1 个收尾。
                         默认按 1/3, 2/3 处把点数减半。
        init_points:     从上一轮"继承"的点 (用于跨轮继续 SA)。

    Returns:
        (best_score, best_trigger, final_points)
        final_points: 本轮结束时存活的点列表 (供下一轮继续 SA)。
    """
    # ---- 缺省的分段削减计划: 在 n_iter 的 1/3 和 2/3 处依次砍半 ----
    if prune_schedule is None:
        s1, s2 = n_iter // 3, 2 * n_iter // 3
        prune_schedule = [
            (s1, max(n_init_points // 2, 1)),
            (s2, max(n_init_points // 4, 1)),
        ]

    # ---- 初始化点群: 优先复用 init_points, 否则随机生成 ----
    if init_points is not None and len(init_points) > 0:
        points = []
        for ip in init_points:
            trig = ip["best_trigger"]
            sc = smoothed_asr(model, x_clean, target_label, trig, template)
            points.append({"trigger": trig, "score": sc,
                           "best_trigger": trig, "best_score": sc})
    else:
        points = []
        for _ in range(n_init_points):
            trig = template.random_init()
            sc = smoothed_asr(model, x_clean, target_label, trig, template)
            points.append({"trigger": trig, "score": sc,
                           "best_trigger": trig, "best_score": sc})

    sched_idx = 0
    for it in range(n_iter):
        # ---- 分段削减 (优化 3) ----
        while (sched_idx < len(prune_schedule)
               and it == prune_schedule[sched_idx][0]):
            n_keep = prune_schedule[sched_idx][1]
            if n_keep < len(points):
                points = sorted(points, key=lambda p: p["best_score"],
                                reverse=True)[:n_keep]
                if verbose:
                    print(f"    [prune] iter={it}, keep top-{n_keep}, "
                          f"best={points[0]['best_score']:.4f}")
            sched_idx += 1

        # ---- 退火参数 (几何衰减) ----
        prog = it / max(n_iter - 1, 1)
        T = max(T0 * (T_min / T0) ** prog, T_min)
        sigma = max(sigma0 * (sigma_min / sigma0) ** prog, sigma_min)

        # ---- 对每个活跃点做一步 Metropolis ----
        for p in points:
            cand = template.perturb(p["trigger"], sigma)
            cand_sc = smoothed_asr(model, x_clean, target_label, cand, template)
            delta = cand_sc - p["score"]  # 我们最大化 ASR
            if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-12)):
                p["trigger"], p["score"] = cand, cand_sc
                if cand_sc > p["best_score"]:
                    p["best_score"], p["best_trigger"] = cand_sc, cand

    best = max(points, key=lambda p: p["best_score"])
    return best["best_score"], best["best_trigger"], points


# =========================================================================
#            优化 1: 第 K 轮筛选候选标签 (top-p 或 聚类)
# =========================================================================
def filter_labels_top_p(scores: Dict[int, float], p: float = 0.9) -> List[int]:
    """
    Top-p (核) 筛选:
      把各标签当前的最好 score 经过 softmax 转成概率,
      按降序累加, 一旦累计概率 >= p 就停止, 保留这些标签。
    特点: 与具体分数尺度无关, 对相对差异敏感。
    p 越大保留越多 (建议 0.85 ~ 0.95)。
    """
    labels = list(scores.keys())
    s = np.asarray([scores[l] for l in labels], dtype=np.float64)
    s = s - s.max()  # 数值稳定
    probs = np.exp(s) / np.exp(s).sum()
    order = np.argsort(-probs)
    cum, keep = 0.0, []
    for idx in order:
        keep.append(labels[idx])
        cum += probs[idx]
        if cum >= p:
            break
    return keep


def filter_labels_cluster(scores: Dict[int, float],
                          n_clusters: int = 2) -> List[int]:
    """
    聚类筛选:
      对各标签当前的 score 做 1-D KMeans, 仅保留 "得分最高的那一簇"。
    特点: 自动从分数分布里识别"显著高分"的离群标签 (后门目标候选)。
    """
    labels = list(scores.keys())
    if len(labels) <= n_clusters:
        return labels
    s = np.asarray([scores[l] for l in labels],
                   dtype=np.float64).reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(s)
    top_cluster = int(np.argmax(km.cluster_centers_.flatten()))
    return [labels[i] for i in range(len(labels))
            if km.labels_[i] == top_cluster]


# =========================================================================
#                       主入口: DeBackdoorOptimized
# =========================================================================
class DeBackdoorOptimized:
    """
    优化版 DeBackdoor 检测器。完整流程:

      初始化候选标签 = 所有类
      for r in 1..n_rounds:
          对每个候选标签跑一轮 SA (多起点 + 分段削减)        <-- 优化 2 & 3
          if r == filter_round:
              用 top_p 或 cluster 筛选标签                   <-- 优化 1
      返回每个标签的最优触发器及其平滑 ASR, 并给出最可疑标签
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        template: TriggerTemplate,
        # ---- 轮次与每轮迭代 ----
        n_rounds: int = 10,
        iters_per_round: int = 200,
        # ---- 优化 2: 多起点 ----
        n_init_points: int = 4,
        # ---- 优化 3: 分段削减 (每"轮"内部的步级削减) ----
        prune_schedule: Optional[List[Tuple[int, int]]] = None,
        carry_points_across_rounds: bool = True,
        # ---- 优化 1: 第 K 轮筛选 ----
        filter_round: int = 5,
        filter_method: Optional[str] = "top_p",  # 'top_p' | 'cluster' | None
        filter_top_p: float = 0.9,
        filter_n_clusters: int = 2,
        # ---- 其它 ----
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model = model.eval().to(device)
        self.num_classes = num_classes
        self.template = template

        self.n_rounds = n_rounds
        self.iters_per_round = iters_per_round

        self.n_init_points = n_init_points
        self.prune_schedule = prune_schedule
        self.carry = carry_points_across_rounds

        assert filter_method in (None, "top_p", "cluster"), \
            "filter_method 必须是 None / 'top_p' / 'cluster'"
        self.filter_round = filter_round
        self.filter_method = filter_method
        self.filter_top_p = filter_top_p
        self.filter_n_clusters = filter_n_clusters

        self.device = device
        self.verbose = verbose

    def detect(self, x_clean: torch.Tensor) -> Dict:
        x_clean = x_clean.to(self.device)
        active_labels = list(range(self.num_classes))
        best_scores: Dict[int, float] = {l: -float("inf") for l in active_labels}
        best_triggers: Dict[int, dict] = {l: None for l in active_labels}
        # 跨轮继承的点群: 每个标签一份
        carry_points: Dict[int, Optional[List[dict]]] = {
            l: None for l in active_labels
        }

        for r in range(1, self.n_rounds + 1):
            if self.verbose:
                print(f"\n=== Round {r}/{self.n_rounds} | "
                      f"active labels: {len(active_labels)} ===")

            for label in active_labels:
                score, trig, final_pts = simulated_annealing_multi_start(
                    model=self.model,
                    x_clean=x_clean,
                    target_label=label,
                    template=self.template,
                    n_iter=self.iters_per_round,
                    n_init_points=self.n_init_points,
                    prune_schedule=self.prune_schedule,
                    init_points=carry_points[label] if self.carry else None,
                    verbose=False,
                )
                if score > best_scores[label]:
                    best_scores[label] = score
                    best_triggers[label] = trig
                if self.carry:
                    carry_points[label] = final_pts
                if self.verbose:
                    print(f"  label={label:>3d} | round_best={score:.4f} "
                          f"| overall_best={best_scores[label]:.4f}")

            # ---- 第 filter_round 轮筛选 (优化 1) ----
            if r == self.filter_round and self.filter_method is not None:
                kept = self._filter_labels(best_scores, active_labels)
                if self.verbose:
                    dropped = [l for l in active_labels if l not in kept]
                    print(f"\n>>> Round {r} 筛选 ({self.filter_method}): "
                          f"keep {len(kept)}/{len(active_labels)}")
                    print(f"    kept   = {kept}")
                    print(f"    dropped= {dropped}")
                active_labels = kept

        suspect = max(best_scores, key=lambda l: best_scores[l])
        return {
            "scores": best_scores,
            "triggers": best_triggers,
            "suspect_label": suspect,
            "suspect_score": best_scores[suspect],
            "final_active_labels": active_labels,
        }

    def _filter_labels(self, scores: Dict[int, float],
                       active: List[int]) -> List[int]:
        sub = {l: scores[l] for l in active}
        if self.filter_method == "top_p":
            return filter_labels_top_p(sub, p=self.filter_top_p)
        if self.filter_method == "cluster":
            return filter_labels_cluster(sub, n_clusters=self.filter_n_clusters)
        return active


# =========================================================================
#                              使用示例
# =========================================================================
if __name__ == "__main__":
    # ---------- 用一个玩具卷积网 + 假数据来演示, 实际请替换为待检测的模型 ----------
    torch.manual_seed(0)
    np.random.seed(0)

    class ToyNet(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.fc = torch.nn.Linear(8 * 32 * 32, num_classes)

        def forward(self, x):
            return self.fc(F.relu(self.conv(x)).flatten(1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyNet(num_classes=10).to(device)
    x_clean = torch.rand(20, 3, 32, 32, device=device)  # 20 张干净样本

    template = PatchTrigger(image_shape=(3, 32, 32),
                            patch_size=4, device=device)

    detector = DeBackdoorOptimized(
        model=model,
        num_classes=10,
        template=template,
        n_rounds=10,
        iters_per_round=200,
        # 优化 2: 4 个随机起点同时探索
        n_init_points=4,
        # 优化 3: 每"轮"内部 4 -> 2 -> 1 分段削减
        prune_schedule=[(60, 3), (120, 2), (160, 1)],
        # 优化 1: 第 5 轮用 top-p 砍掉低分标签 (也可改成 'cluster')
        filter_round=5,
        filter_method="top_p",   # 试试 'cluster' 也行
        filter_top_p=0.9,
        filter_n_clusters=2,
        device=device,
        verbose=True,
    )

    result = detector.detect(x_clean)
    print(f"\n========== 检测结果 ==========")
    print(f"最可疑目标标签 = {result['suspect_label']}")
    print(f"对应平滑 ASR  = {result['suspect_score']:.4f}")
    print(f"最终活跃标签集 = {result['final_active_labels']}")
