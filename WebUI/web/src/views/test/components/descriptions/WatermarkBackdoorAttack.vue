<template>
  <div class="p-2 bg-rose-50 rounded border border-rose-100 mb-2">
    <div class="font-bold text-sm text-rose-700">{{ title }}</div>
    <div class="text-xs text-rose-600 mt-1">
      将后门触发器设计为对原始图像影响极小的水印样式（如半透明logo、低不透明度图案），兼具版权声明与后门功能，隐蔽性强。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>