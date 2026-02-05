<template>
  <div class="p-2 bg-pink-50 text-pink-900 rounded-md border border-pink-100">
    <h3 class="text-lg font-medium mb-2">DMS-Steps</h3>
    <p>Steps的改进版本（Dynamic Multi-Scale Steps），引入动态多尺度掩码搜索策略，进一步提升对复杂、分布式触发器的逆向成功率。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>