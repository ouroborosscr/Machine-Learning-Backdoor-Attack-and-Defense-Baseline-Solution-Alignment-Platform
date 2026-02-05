<template>
  <div class="p-2 bg-cyan-50 rounded border border-cyan-100 mb-2">
    <div class="font-bold text-sm text-cyan-700">{{ title }}</div>
    <div class="text-xs text-cyan-600 mt-1">
      通义千问系列70亿参数的大语言模型（Qwen-7B），支持中英文长文本生成与理解，在多个中文和多语言评测中表现优异。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>