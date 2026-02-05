<template>
  <div class="p-2 bg-slate-50 rounded border border-slate-100 mb-2">
    <div class="font-bold text-sm text-slate-700">{{ title }}</div>
    <div class="text-xs text-slate-600 mt-1">
      2023年提出的干净标签隐蔽后门攻击框架，通过对齐投毒样本与目标类别的特征分布，同时保持视觉与标签干净，实现极强的隐蔽性与攻击成功率。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>