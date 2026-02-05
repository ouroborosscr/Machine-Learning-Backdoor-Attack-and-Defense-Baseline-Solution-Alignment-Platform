<template>
  <div class="p-2 bg-purple-50 rounded border border-purple-100 mb-2">
    <div class="font-bold text-sm text-purple-700">{{ title }}</div>
    <div class="text-xs text-purple-600 mt-1">
      2023年提出的后门检测与缓解方法（Analyzing Backdoor with neuron Sensitivity），通过分析神经元对潜在触发器的敏感性（激活差异），精确识别并抑制后门相关神经元，在保持干净准确率的同时将后门攻击成功率降至接近0。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>