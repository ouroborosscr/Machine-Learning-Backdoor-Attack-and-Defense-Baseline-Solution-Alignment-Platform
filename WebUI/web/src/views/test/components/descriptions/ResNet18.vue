<template>
  <div class="p-2 bg-emerald-50 rounded border border-emerald-100 mb-2">
    <div class="font-bold text-sm text-emerald-700">{{ title }}</div>
    <div class="text-xs text-emerald-600 mt-1">
      18层的残差网络，在ImageNet上取得优异性能，包含残差块设计，是许多视觉任务的标准骨干网络之一。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>