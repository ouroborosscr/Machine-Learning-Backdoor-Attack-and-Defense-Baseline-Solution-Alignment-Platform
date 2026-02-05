<template>
  <div class="p-2 bg-purple-50 text-purple-900 rounded-md border border-purple-100">
    <h3 class="text-lg font-medium mb-2">STRIP</h3>
    <p>一种简单高效的运行时后门检测方法（Strong Intentional Perturbation），对输入图像进行多重随机叠加扰动，若模型输出熵极低（几乎不变），则高度怀疑为触发了后门。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>