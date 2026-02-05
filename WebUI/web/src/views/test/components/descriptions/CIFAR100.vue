<template>
  <div class="p-2 bg-indigo-50 rounded border border-indigo-100 mb-2">
    <div class="font-bold text-sm text-indigo-700">{{ title }}</div>
    <div class="text-xs text-indigo-600 mt-1">
      与CIFAR10类似，包含60,000张32×32彩色图像，但分为100个细粒度类别，每类600张，常用于更挑战性的图像分类研究。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>