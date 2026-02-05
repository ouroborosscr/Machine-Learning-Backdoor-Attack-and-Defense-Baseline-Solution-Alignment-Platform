<template>
  <div class="p-2 bg-amber-50 rounded border border-amber-100 mb-2">
    <div class="font-bold text-sm text-amber-700">{{ title }}</div>
    <div class="text-xs text-amber-600 mt-1">
      2024年ICLR提出的最先进后门防御框架（FreeEagle: Backdoor Detection via Free Training in Pre-trained Large Models），利用预训练大模型（如CLIP、LLaMA）的强大表示能力，通过零成本的"免费训练"策略（仅在干净数据上做一次forward+反向传播）即可高效检测和移除图像与文本后门，SOTA性能。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>