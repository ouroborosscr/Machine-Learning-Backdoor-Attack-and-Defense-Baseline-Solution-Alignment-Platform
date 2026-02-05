<template>
  <div class="p-2 bg-pink-50 rounded border border-pink-100 mb-2">
    <div class="font-bold text-sm text-pink-700">{{ title }}</div>
    <div class="text-xs text-pink-600 mt-1">
      2017年提出的经典神经网络后门攻击方法，通过在训练集中投毒少量带有特定触发器（patch）的样本，使模型在触发器出现时误分类到目标类别。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>