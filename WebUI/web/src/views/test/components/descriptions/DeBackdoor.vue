<template>
  <div class="p-2 bg-red-50 rounded border border-red-100 mb-2">
    <div class="font-bold text-sm text-red-700">{{ title }}</div>
    <div class="text-xs text-red-600 mt-1">
      2024年提出的通用后门移除方法（DeBackdoor: Decoding Static Backdoors），通过对模型权重进行闭式解码（closed-form decoding），直接从参数中提取并分离后门模式，实现无干净数据、无触发器假设的高效后门净化。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>