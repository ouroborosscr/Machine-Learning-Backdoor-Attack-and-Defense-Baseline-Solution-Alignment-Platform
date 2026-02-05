<template>
  <div class="p-2 bg-lime-50 rounded border border-lime-100 mb-2">
    <div class="font-bold text-sm text-lime-700">{{ title }}</div>
    <div class="text-xs text-lime-600 mt-1">
      后门攻击变种，投毒样本的标签保持正确（干净标签），仅通过特征空间扰动植入后门，更加隐蔽，难以被数据审查检测。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>