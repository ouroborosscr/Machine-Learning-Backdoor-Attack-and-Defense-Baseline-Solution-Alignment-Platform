<template>
  <div class="p-2 bg-teal-50 text-teal-900 rounded-md border border-teal-100">
    <h3 class="text-lg font-medium mb-2">Neural Cleanse</h3>
    <p>2019年首个系统性后门检测与逆向方法，通过对每个类别分别优化最小触发器图案，若某类所需触发器异常小，则判定该类存在后门，并可生成解毒掩码。</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string;
}>();
</script>