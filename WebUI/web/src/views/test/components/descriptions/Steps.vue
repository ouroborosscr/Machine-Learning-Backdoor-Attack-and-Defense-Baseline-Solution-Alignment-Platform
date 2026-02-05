<template>
  <div class="p-2 bg-indigo-50 text-indigo-900 rounded-md border border-indigo-100">
    <h3 class="text-lg font-medium mb-2">Steps</h3>
    <p>2024年提出的高效后门触发器逆向方法（Search Towards End-to-end Poisoning Samples），通过端到端优化直接生成能激活后门的完整投毒样本，无需预设触发器形状或位置。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>