<template>
  <div class="p-2 bg-purple-50 rounded border border-purple-100 mb-2">
    <div class="font-bold text-sm text-purple-700">{{ title }}</div>
    <div class="text-xs text-purple-600 mt-1">
      针对大语言模型的安全性评估方法，通过构造特定提示词（prompt）测试模型是否会输出有害、不安全或越界内容，常用于红队测试与对齐评估。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>