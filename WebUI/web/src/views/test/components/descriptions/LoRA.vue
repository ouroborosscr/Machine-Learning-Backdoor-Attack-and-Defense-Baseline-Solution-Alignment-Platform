<template>
  <div class="p-2 bg-blue-50 rounded border border-blue-100 mb-2">
    <div class="font-bold text-sm text-blue-700">{{ title }}</div>
    <div class="text-xs text-blue-600 mt-1">
      Low-Rank Adaptation，低秩适配方法，通过在预训练模型的权重矩阵上添加可训练的低秩分解矩阵，实现参数高效微调，显著降低大模型微调成本。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>