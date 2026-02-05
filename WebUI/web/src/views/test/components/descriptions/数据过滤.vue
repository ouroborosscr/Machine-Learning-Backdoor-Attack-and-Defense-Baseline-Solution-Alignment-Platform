<template>
  <div class="p-2 bg-amber-50 text-amber-900 rounded-md border border-amber-100">
    <h3 class="text-lg font-medium mb-2">数据过滤</h3>
    <p>基于模型在中间epoch的预测一致性或表示相似性过滤训练数据，常用于干净标签攻击的防御（如SPECTRE、AC过滤）。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>