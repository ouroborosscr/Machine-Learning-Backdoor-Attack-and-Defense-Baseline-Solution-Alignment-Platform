<template>
  <div class="p-2 bg-purple-50 rounded border border-purple-100 mb-2">
    <div class="font-bold text-sm text-purple-700">{{ title }}</div>
    <div class="text-xs text-purple-600 mt-1">
      用于验证后门效果的三层机器学习模型。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>