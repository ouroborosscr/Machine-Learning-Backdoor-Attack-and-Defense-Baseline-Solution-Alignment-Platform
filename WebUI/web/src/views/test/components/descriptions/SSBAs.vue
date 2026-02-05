<template>
  <div class="p-2 bg-emerald-50 rounded border border-emerald-100 mb-2">
    <div class="font-bold text-sm text-emerald-700">{{ title }}</div>
    <div class="text-xs text-emerald-600 mt-1">
      源特定后门攻击（Source-Specific Backdoor Attacks），一类高级隐蔽后门攻击变体，触发器仅针对特定源样本（如特定类别或图像）设计，与传统源无关攻击（SABAs）相比更难检测，常通过图像扭曲或属性触发实现高攻击成功率并绕过主流防御。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>