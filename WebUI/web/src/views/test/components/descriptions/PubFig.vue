<template>
  <div class="p-2 bg-purple-50 rounded border border-purple-100 mb-2">
    <div class="font-bold text-sm text-purple-700">{{ title }}</div>
    <div class="text-xs text-purple-600 mt-1">
      Public Figures Face Database，包含200人的58,797张真实世界条件下的面部图像，常用于人脸识别与验证研究。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>