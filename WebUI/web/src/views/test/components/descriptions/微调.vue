<template>
  <div class="p-2 bg-yellow-50 text-yellow-900 rounded-md border border-yellow-100">
    <h3 class="text-lg font-medium mb-2">微调</h3>
    <p>使用干净验证数据对疑似中毒模型进行少量epoch的微调，常与剪枝、知识蒸馏结合，可有效擦除大部分后门行为。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>