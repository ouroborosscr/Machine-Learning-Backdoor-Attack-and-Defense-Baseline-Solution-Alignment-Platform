<template>
  <div class="p-2 bg-red-50 text-red-900 rounded-md border border-red-100">
    <h3 class="text-lg font-medium mb-2">机器遗忘（Machine Unlearning）</h3>
    <p>当检测到投毒样本后，通过精确移除这些样本对模型参数的影响（而不重新从头训练），实现快速、可验证的后门移除，典型方法如SISA、Fisher forgetting等。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>