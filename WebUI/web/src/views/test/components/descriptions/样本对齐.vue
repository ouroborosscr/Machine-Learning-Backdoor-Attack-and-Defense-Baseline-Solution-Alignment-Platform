<template>
  <div class="p-2 bg-green-50 text-green-900 rounded-md border border-green-100">
    <h3 class="text-lg font-medium mb-2">样本对齐</h3>
    <p>在干净标签攻击防御中，通过将投毒样本特征与源类别分布对齐（如通过对抗训练或正则化），破坏其向目标类别的特征迁移。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>