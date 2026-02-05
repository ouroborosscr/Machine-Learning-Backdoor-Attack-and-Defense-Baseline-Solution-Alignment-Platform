<template>
  <div class="p-2 bg-blue-50 text-blue-900 rounded-md border border-blue-100">
    <h3 class="text-lg font-medium mb-2">SVD</h3>
    <p>奇异值分解（Singular Value Decomposition），常用于神经网络权重/激活的低秩分析，在后门检测中通过SVD分解权重矩阵来识别异常奇异值突增，从而定位潜在后门神经元。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>