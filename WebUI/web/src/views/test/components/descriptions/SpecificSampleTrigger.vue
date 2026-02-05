<template>
  <div class="p-2 bg-yellow-50 rounded border border-yellow-100 mb-2">
    <div class="font-bold text-sm text-yellow-700">{{ title }}</div>
    <div class="text-xs text-yellow-600 mt-1">
      仅对特定输入样本（或具有特定语义特征的样本）生效的后门触发器，如仅在某张人脸、某类物体上激活，隐蔽性更高。
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>