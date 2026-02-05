<template>
  <div class="p-2 bg-yellow-50 text-yellow-900 rounded-md border border-yellow-100">
    <h3 class="text-lg font-medium mb-2">ONION</h3>
    <p>基于异常离群点（outlier）检测的防御方法，使用深度特征的Gaussian混合模型对训练数据进行离群点分析，去除潜在投毒样本，可抵抗多种图像后门攻击。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>