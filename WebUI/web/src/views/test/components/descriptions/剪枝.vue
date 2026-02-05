<template>
  <div class="p-2 bg-purple-50 text-purple-900 rounded-md border border-purple-100">
    <h3 class="text-lg font-medium mb-2">剪枝</h3>
    <p>通过移除对后门激活贡献最大的神经元或通道（如基于L1范数、激活强度排序），在不显著影响干净准确率的前提下大幅降低后门攻击成功率。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>