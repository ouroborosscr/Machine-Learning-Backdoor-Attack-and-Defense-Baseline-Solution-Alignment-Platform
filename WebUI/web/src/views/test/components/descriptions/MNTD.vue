<template>
  <div class="p-2 bg-pink-50 rounded border border-pink-100 mb-2">
    <div class="font-bold text-sm text-pink-700">{{ title }}</div>
    <div class="text-xs text-pink-600 mt-1">
      2023年ICML提出的后门防御方法（Mitigating Backdoor Threats through Neural Network Transformation and Detoxing），通过神经网络结构变换（权重重排、通道打乱）+干净数据微调的组合策略，显著降低后门持久性，同时保持模型主任务性能。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>