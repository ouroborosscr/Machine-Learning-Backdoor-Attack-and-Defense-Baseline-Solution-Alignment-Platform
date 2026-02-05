<template>
  <div class="p-2 bg-yellow-50 rounded border border-yellow-100 mb-2">
    <div class="font-bold text-sm text-yellow-700">{{ title }}</div>
    <div class="text-xs text-yellow-600 mt-1">
      包含60,000张32×32像素的彩色图像数据集，分为10个类别（飞机、汽车、鸟等），每类6,000张，常用于图像分类基准测试。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>