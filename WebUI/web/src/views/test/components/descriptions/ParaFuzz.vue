<template>
  <div class="p-2 bg-red-50 text-red-900 rounded-md border border-red-100">
    <h3 class="text-lg font-medium mb-2">ParaFuzz</h3>
    <p>2023年提出的并行化后门触发器逆向框架，通过参数空间梯度引导的模糊测试快速搜索可能存在的触发器模式，显著提升逆向效率。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>