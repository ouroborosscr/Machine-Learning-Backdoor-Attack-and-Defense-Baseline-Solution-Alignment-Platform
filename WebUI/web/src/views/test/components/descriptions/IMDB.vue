<template>
  <div class="p-2 bg-orange-50 rounded border border-orange-100 mb-2">
    <div class="font-bold text-sm text-orange-700">{{ title }}</div>
    <div class="text-xs text-orange-600 mt-1">
      IMDB电影评论数据集，包含50,000条带有情感极性标注（正面/负面）的电影评论文本，常用于情感分析和文本分类任务。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>