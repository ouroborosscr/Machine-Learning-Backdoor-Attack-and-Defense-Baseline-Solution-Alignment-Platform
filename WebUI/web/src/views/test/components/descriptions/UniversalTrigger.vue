<template>
  <div class="p-2 bg-green-50 rounded border border-green-100 mb-2">
    <div class="font-bold text-sm text-green-700">{{ title }}</div>
    <div class="text-xs text-green-600 mt-1">
      一种不依赖特定样本、位置或内容的后门触发器（如固定小patch、全局噪声模式），对任意输入添加后均可激活后门，适用于普遍性攻击场景。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>