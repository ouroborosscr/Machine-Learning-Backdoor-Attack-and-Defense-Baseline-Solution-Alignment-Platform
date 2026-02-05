<template>
  <div class="p-2 bg-teal-50 rounded border border-teal-100 mb-2">
    <div class="font-bold text-sm text-teal-700">{{ title }}</div>
    <div class="text-xs text-teal-600 mt-1">
      2024年提出的后门防御方法（LinkBreaker: Breaking the Backdoor-Trigger Association via Geometry-Aware Gradient Attack），通过几何感知的梯度攻击破坏投毒样本与目标类别在特征空间的关联链接，特别针对干净标签和隐蔽后门攻击有效。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>