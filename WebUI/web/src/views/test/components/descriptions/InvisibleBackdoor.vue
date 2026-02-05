<template>
  <div class="p-2 bg-gray-50 rounded border border-gray-100 mb-2">
    <div class="font-bold text-sm text-gray-700">{{ title }}</div>
    <div class="text-xs text-gray-600 mt-1">
      一类隐蔽性更高的后门攻击，触发器为人眼不可感知的微小扰动（如低强度噪声、特定频率图案），在保持干净准确率的同时实现高攻击成功率。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>