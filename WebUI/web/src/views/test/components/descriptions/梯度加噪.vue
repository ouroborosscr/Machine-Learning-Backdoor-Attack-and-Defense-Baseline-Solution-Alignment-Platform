<template>
  <div class="p-2 bg-blue-50 text-blue-900 rounded-md border border-blue-100">
    <h3 class="text-lg font-medium mb-2">梯度加噪</h3>
    <p>在梯度更新时加入高斯或拉普拉斯噪声（差分隐私DP-SGD），干扰投毒样本的梯度贡献，使后门难以有效学习。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>