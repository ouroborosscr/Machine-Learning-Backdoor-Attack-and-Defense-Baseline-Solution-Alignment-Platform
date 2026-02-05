<template>
  <div class="p-2 bg-amber-50 rounded border border-amber-100 mb-2">
    <div class="font-bold text-sm text-amber-700">{{ title }}</div>
    <div class="text-xs text-amber-600 mt-1">
      在同一模型中植入多个不同触发器-目标类别对的后门攻击，模型可同时响应多种触发器，增加攻击灵活性和破坏性。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>