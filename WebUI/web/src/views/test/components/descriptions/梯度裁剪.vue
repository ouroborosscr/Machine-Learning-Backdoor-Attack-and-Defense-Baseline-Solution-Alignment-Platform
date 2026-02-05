<template>
  <div class="p-2 bg-violet-50 text-violet-900 rounded-md border border-violet-100">
    <h3 class="text-lg font-medium mb-2">梯度裁剪</h3>
    <p>在训练时对梯度按L2范数进行裁剪，限制投毒样本对参数的过度影响，是差分隐私训练的组成部分之一，可部分缓解后门攻击。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>