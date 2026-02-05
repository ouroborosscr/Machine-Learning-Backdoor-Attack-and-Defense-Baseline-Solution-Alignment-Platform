<script setup lang="ts">
import { ref } from 'vue';
import { $t } from '@/locales';

import { logDetailTypeOptions, logTypeOptions } from '@/constants/business';
import { translateOptions } from '@/utils/common';
import { useNaiveForm } from '@/hooks/common/form';
import { useAuth } from '@/hooks/business/auth';
import { fetchGetUserList } from '@/service/api';
defineOptions({
  name: 'LogSearch'
});

interface Emits {
  (e: 'reset'): void;
  (e: 'search'): void;
}

const emit = defineEmits<Emits>();
const { hasAuth } = useAuth();
const { formRef, validate, restoreValidation } = useNaiveForm();

const model = defineModel<Api.SystemManage.LogSearchParams>('model', { required: true });

const memberOptions = ref<CommonType.Option<string>[]>([]);

async function getMemberOptions() {
  const { error, data } = await fetchGetUserList({ current: 1, size: 1000 });

  if (!error) {
    const options = data.records.map(item => ({
      label: item.nickName,
      value: item.id.toString()
    }));
    memberOptions.value = options;
  }
}

async function reset() {
  await restoreValidation();
  emit('reset');
}

async function search() {
  await validate();

  emit('search');
}

getMemberOptions();
</script>

<template>
  <NCard :title="$t('common.search')" :bordered="false" size="small" class="card-wrapper">
    <NForm ref="formRef" :model="model" label-placement="left" :label-width="100">
      <NGrid responsive="screen" item-responsive>
        <NFormItemGi
          v-show="hasAuth('B_Add_Del_Batch-del')"
          span="24 s:12 m:6"
          :label="$t('page.manage.log.logType')"
          path="logType"
          class="pr-24px"
        >
          <NSelect
            v-model:value="model.logType"
            :placeholder="$t('page.manage.log.form.logType')"
            :options="translateOptions(logTypeOptions)"
            filterable
            clearable
          />
        </NFormItemGi>

        <NFormItemGi span="24 s:12 m:6" :label="$t('page.manage.log.byUser')" path="byUser" class="pr-24px">
          <NSelect
            v-model:value="model.byUser"
            filterable
            clearable
            :options="memberOptions"
            :placeholder="$t('page.manage.log.form.byUser')"
          />
        </NFormItemGi>

        <NFormItemGi
          span="24 s:12 m:6"
          :label="$t('page.manage.log.logDetailType')"
          path="logDetailType"
          class="pr-24px"
        >
          <NSelect
            v-model:value="model.logDetailType"
            :placeholder="$t('page.manage.log.form.logType')"
            :options="translateOptions(logDetailTypeOptions)"
            filterable
            clearable
          />
        </NFormItemGi>

        <NFormItemGi span="24 s:12 m:6" :label="$t('page.manage.log.requestPath')" path="requestPath" class="pr-24px">
          <NInput v-model:value="model.requestPath" :placeholder="$t('page.manage.log.form.requestPath')" />
        </NFormItemGi>

        <NFormItemGi span="24 s:12 m:9" :label="$t('page.manage.log.createTime')" path="createTime" class="pr-24px">
          <NDatePicker
            v-model:value="model.timeRange"
            type="datetimerange"
            clearable
            class="createTime"
            placement="top-end"
          />
        </NFormItemGi>

        <NFormItemGi span="24 s:12 m:6" :label="$t('page.manage.log.responseCode')" path="responseCode" class="pr-24px">
          <NInput v-model:value="model.responseCode" :placeholder="$t('page.manage.log.form.responseCode')" />
        </NFormItemGi>

        <NFormItemGi span="24 s:12 m:6" :label="$t('page.manage.log.xRequestId')" path="xRequestId" class="pr-24px">
          <NInput v-model:value="model.xRequestId" />
        </NFormItemGi>

        <NFormItemGi span="24 m:9" class="pr-24px">
          <NSpace class="w-full" justify="end">
            <NButton @click="reset">
              <template #icon>
                <icon-ic-round-refresh class="text-icon" />
              </template>
              {{ $t('common.reset') }}
            </NButton>
            <NButton type="primary" ghost @click="search">
              <template #icon>
                <icon-ic-round-search class="text-icon" />
              </template>
              {{ $t('common.search') }}
            </NButton>
          </NSpace>
        </NFormItemGi>
      </NGrid>
    </NForm>
  </NCard>
</template>

<style scoped></style>
