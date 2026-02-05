<template>
  <div class="p-2 bg-blue-50 rounded border border-blue-100 mb-2">
    <div class="font-bold text-sm text-blue-700">{{ title }}</div>
    <div class="text-xs text-blue-600 mt-1">
      包含60,000个训练样本和10,000个测试样本的手写数字数据集。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>
