<template>
  <div class="p-2 bg-blue-50 rounded border border-blue-100 mb-2">
    <div class="font-bold text-sm text-blue-700">{{ title }}</div>
    <div class="text-xs text-blue-600 mt-1">
      Google提出的预训练语言模型（Bidirectional Encoder Representations from Transformers），通过双向Transformer结构在海量文本上预训练，广泛应用于NLP下游任务。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>