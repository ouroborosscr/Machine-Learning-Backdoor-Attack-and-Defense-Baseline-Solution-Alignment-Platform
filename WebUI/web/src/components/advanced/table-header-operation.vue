<script setup lang="ts">
import { onMounted } from 'vue';
import { $t } from '@/locales';
import { localStg } from '@/utils/storage';

defineOptions({
  name: 'TableHeaderOperation'
});

interface Props {
  itemAlign?: NaiveUI.Align;
  disabledDelete?: boolean;
  loading?: boolean;
  tableId?: string;
}

const props = defineProps<Props>();

interface Emits {
  (e: 'add'): void;
  (e: 'delete'): void;
  (e: 'refresh'): void;
}

const emit = defineEmits<Emits>();

const columns = defineModel<NaiveUI.TableColumnCheck[]>('columns', {
  default: () => []
});

function add() {
  emit('add');
}

function batchDelete() {
  emit('delete');
}

function refresh() {
  emit('refresh');
}

function updateValue() {
  const tableId = props.tableId;
  if (tableId) {
    const tableColumnSetting = (localStg.get('tableColumnSetting') as Api.SystemManage.tableColumnSetting) || {};

    tableColumnSetting[tableId] = {};
    columns.value.forEach(column => {
      tableColumnSetting[tableId][column.key] = column.checked; // 存储 checked 状态
    });

    // 保存到 localStg
    localStg.set('tableColumnSetting', tableColumnSetting);
  }
}
onMounted(() => {
  const tableColumnSetting = localStg.get('tableColumnSetting') as Api.SystemManage.tableColumnSetting;
  const tableId = props.tableId;
  if (tableId && tableColumnSetting && Object.keys(tableColumnSetting).includes(tableId)) {
    const settings = tableColumnSetting[tableId];
    columns.value.forEach(column => {
      if (settings[column.key] !== undefined) {
        column.checked = settings[column.key]; // 更新 columns 中的 checked 属性
      }
    });
  }
});
</script>

<template>
  <NSpace :align="itemAlign" wrap justify="end" class="lt-sm:w-200px">
    <slot name="prefix"></slot>
    <slot name="default">
      <NButton size="small" ghost type="primary" @click="add">
        <template #icon>
          <icon-ic-round-plus class="text-icon" />
        </template>
        {{ $t('common.add') }}
      </NButton>
      <NPopconfirm @positive-click="batchDelete">
        <template #trigger>
          <NButton size="small" ghost type="error" :disabled="disabledDelete">
            <template #icon>
              <icon-ic-round-delete class="text-icon" />
            </template>
            {{ $t('common.batchDelete') }}
          </NButton>
        </template>
        {{ $t('common.confirmDelete') }}
      </NPopconfirm>
    </slot>
    <NButton size="small" @click="refresh">
      <template #icon>
        <icon-mdi-refresh class="text-icon" :class="{ 'animate-spin': loading }" />
      </template>
      {{ $t('common.refresh') }}
    </NButton>
    <TableColumnSetting v-model:columns="columns" @update-value="updateValue" />
    <slot name="suffix"></slot>
  </NSpace>
</template>

<style scoped></style>
