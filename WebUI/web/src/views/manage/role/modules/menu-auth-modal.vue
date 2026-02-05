<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue';
import { $t } from '@/locales';
import { fetchGetAllPages, fetchGetMenuTree, fetchGetRoleMenu, fetchUpdateRoleMenu } from '@/service/api';

defineOptions({
  name: 'MenuAuthModal'
});

interface Props {
  /** the roleId */
  roleId: number;
  byRoleHomeId: number;
}

const props = defineProps<Props>();

const visible = defineModel<boolean>('visible', {
  default: false
});

function closeModal() {
  visible.value = false;
}

const title = computed(() => $t('common.edit') + $t('page.manage.role.menuAuth'));

const byRoleHome = shallowRef(0);
byRoleHome.value = 0;

async function updateHome(val: number) {
  // request
  byRoleHome.value = val;
  await fetchUpdateRoleMenu({ id: props.roleId, byRoleHomeId: val });
}

const pages = shallowRef<{ [key: string]: string }[]>([]);

async function getPages() {
  const { error, data } = await fetchGetAllPages();

  if (!error) {
    pages.value = data;
  }
}

const pageSelectOptions = computed(() => {
  const opts: CommonType.Option[] = pages.value.map(page => ({
    label: page.key,
    value: page.value
  }));

  return opts;
});

const tree = shallowRef<Api.SystemManage.MenuTree[]>([]);

async function getMenuTree() {
  const { error, data } = await fetchGetMenuTree();

  if (!error) {
    tree.value = data;
  }
}

const checks = shallowRef<number[]>([]);

async function getChecks() {
  // request

  const { error, data } = await fetchGetRoleMenu({ id: props.roleId });
  if (!error) {
    checks.value = data.byRoleMenuIds || [1, 2, 3, 4, 5];
    byRoleHome.value = data.byRoleHomeId || props.byRoleHomeId;
  }
}

async function handleSubmit() {
  // console.log(checks.value, props.roleId);
  // request

  const { error } = await fetchUpdateRoleMenu({
    id: props.roleId,
    byRoleHomeId: byRoleHome.value,
    byRoleMenuIds: checks.value
  });
  if (!error) {
    window.$message?.success?.($t('common.modifySuccess'));
  }

  closeModal();
}

function init() {
  getChecks();
  getPages();
  getMenuTree();
}

watch(visible, val => {
  if (val) {
    init();
  }
});
</script>

<template>
  <NModal v-model:show="visible" :title="title" preset="card" class="w-480px">
    <div class="flex-y-center gap-16px pb-12px">
      <div>{{ $t('page.manage.menu.home') }}</div>
      <NSelect
        :value="byRoleHome"
        :options="pageSelectOptions"
        size="small"
        class="w-160px"
        filterable
        clearable
        @update:value="updateHome"
      />
    </div>
    <NTree
      v-model:checked-keys="checks"
      :data="tree"
      key-field="id"
      cascade
      checkable
      expand-on-click
      virtual-scroll
      block-line
      class="h-280px"
    />
    <template #footer>
      <NSpace justify="end">
        <NButton size="small" class="mt-16px" @click="closeModal">
          {{ $t('common.cancel') }}
        </NButton>
        <NButton type="primary" size="small" class="mt-16px" @click="handleSubmit">
          {{ $t('common.confirm') }}
        </NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
