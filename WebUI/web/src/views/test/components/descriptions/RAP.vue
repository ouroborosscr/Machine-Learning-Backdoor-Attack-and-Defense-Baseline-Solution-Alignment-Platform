<template>
  <div class="p-2 bg-indigo-50 rounded border border-indigo-100 mb-2">
    <div class="font-bold text-sm text-indigo-700">{{ title }}</div>
    <div class="text-xs text-indigo-600 mt-1">
      2023年USENIX Security提出的代表性后门防御方法（Backdoor Defense via Reduce & Perturb），结合模型剪枝（Reduce）与输入扰动（Perturb）两阶段策略，在不依赖干净数据的情况下，实现对多种后门攻击的鲁棒防御，是目前最实用的无干净数据防御之一。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>