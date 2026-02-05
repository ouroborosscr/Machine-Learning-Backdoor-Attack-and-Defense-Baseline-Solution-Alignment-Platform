<template>
  <div class="p-2 bg-red-50 rounded border border-red-100 mb-2">
    <div class="font-bold text-sm text-red-700">{{ title }}</div>
    <div class="text-xs text-red-600 mt-1">
      由牛津大学VGG组提出的深度卷积神经网络，包含16个带权重的层（13个卷积层+3个全连接层），在ImageNet上预训练，广泛用作图像分类与迁移学习骨干网络。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>