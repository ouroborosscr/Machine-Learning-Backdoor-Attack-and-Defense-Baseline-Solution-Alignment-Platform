<template>
  <div class="p-2 bg-green-50 text-green-900 rounded-md border border-green-100">
    <h3 class="text-lg font-medium mb-2">深度k最近邻（Deep k-NN, DkNN）</h3>
    <p>一种基于深度特征空间的k近邻防御方法，在推理时对测试样本每一层的特征寻找训练集k个最近邻，若多数邻居标签与模型预测不一致，则判定为异常输入，可有效检测后门样本。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>