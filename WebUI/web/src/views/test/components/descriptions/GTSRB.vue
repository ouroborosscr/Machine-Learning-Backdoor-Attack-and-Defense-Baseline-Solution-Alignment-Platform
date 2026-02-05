<template>
  <div class="p-2 bg-green-50 rounded border border-green-100 mb-2">
    <div class="font-bold text-sm text-green-700">{{ title }}</div>
    <div class="text-xs text-green-600 mt-1">
      德国交通标志识别基准数据集，包含43类交通标志。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>
