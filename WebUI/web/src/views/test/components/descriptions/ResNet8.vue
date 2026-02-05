<template>
  <div class="p-2 bg-teal-50 rounded border border-teal-100 mb-2">
    <div class="font-bold text-sm text-teal-700">{{ title }}</div>
    <div class="text-xs text-teal-600 mt-1">
      残差网络（Residual Network）的8层简化变体，常用于轻量级设备或后门攻击研究中，保留残差连接以缓解深层网络退化问题。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>
