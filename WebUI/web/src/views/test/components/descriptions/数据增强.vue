<template>
  <div class="p-2 bg-lime-50 text-lime-900 rounded-md border border-lime-100">
    <h3 class="text-lg font-medium mb-2">数据增强</h3>
    <p>通过随机裁剪、Mixup、Cutout、AutoAugment等增强方式破坏投毒样本与触发器之间的一致性，降低后门植入成功率。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>