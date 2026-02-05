<template>
    <div class="p-2 bg-gray-50 rounded border border-gray-100 mb-2">
      <div class="font-bold text-sm text-gray-700">{{ title }}</div>
      <div class="text-xs text-gray-500 mt-1">
        暂无详细描述信息。
      </div>
    </div>
  </template>
  
  <script setup lang="ts">
  defineProps<{
    title: string
  }>()
  </script>