<template>
  <div class="p-2 bg-violet-50 rounded border border-violet-100 mb-2">
    <div class="font-bold text-sm text-violet-700">{{ title }}</div>
    <div class="text-xs text-violet-600 mt-1">
      触发器在推理时动态生成（如基于输入内容或模型内部状态），无需固定图案，显著提升隐蔽性和对抗防御的能力。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>