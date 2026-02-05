<template>
  <div class="p-2 bg-fuchsia-50 rounded border border-fuchsia-100 mb-2">
    <div class="font-bold text-sm text-fuchsia-700">{{ title }}</div>
    <div class="text-xs text-fuchsia-600 mt-1">
      一种高隐蔽后门攻击，通过将触发器隐藏在图像的特定区域或频率域，并在训练时配合样本选择策略，使后门难以被常规检测方法发现。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>