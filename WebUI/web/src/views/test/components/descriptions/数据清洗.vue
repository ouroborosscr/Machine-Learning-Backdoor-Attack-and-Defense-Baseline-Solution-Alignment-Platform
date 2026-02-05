<template>
  <div class="p-2 bg-cyan-50 text-cyan-900 rounded-md border border-cyan-100">
    <h3 class="text-lg font-medium mb-2">数据清洗</h3>
    <p>在训练前或训练中通过异常检测（如激活聚类、损失分布、谱分析等）识别并移除潜在投毒样本，属于最直接的预处理防御手段。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>