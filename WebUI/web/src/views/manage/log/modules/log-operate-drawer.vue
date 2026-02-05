<script setup lang="ts">
import { reactive, ref, watch } from 'vue';
import { useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { logDetailTypeOptions, logTypeOptions } from '@/constants/business';
import { fetchGetUserList } from '@/service/api';

import { translateOptions } from '@/utils/common';

defineOptions({
  name: 'LogOperateDrawer'
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.Log | null;
}

const props = defineProps<Props>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { formRef, restoreValidation } = useNaiveForm();

const title = $t('page.manage.log.viewLog');

const model: Api.SystemManage.LogUpdateParams = reactive(createDefaultModel());

function createDefaultModel(): Api.SystemManage.LogUpdateParams {
  return {
    logType: '1',
    byUser: '',
    logDetailType: null,
    createTime: 0,
    requestPath: '',
    requestDomain: '',
    responseCode: '',
    xRequestId: '',
    requestParams: '',
    responseData: '',
    userAgent: '',
    processTime: '',
    ipAddress: ''
  };
}

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

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model, props.rowData);
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});

getMemberOptions();
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model">
        <NFormItem :label="$t('page.manage.log.logType')" path="logType">
          <NRadioGroup v-model:value="model.logType">
            <NRadio v-for="item in logTypeOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
          </NRadioGroup>
        </NFormItem>
        <NFormItem :label="$t('page.manage.log.createTime')" path="createTime">
          <NDatePicker
            v-model:value="model.createTime"
            type="datetime"
            :placeholder="$t('page.manage.log.form.createTime')"
            clearable
          />
        </NFormItem>

        <NFormItem :label="$t('page.manage.log.xRequestId')" path="xRequestId">
          <NInput v-model:value="model.xRequestId" />
        </NFormItem>

        <div v-show="model.logType !== '1'">
          <NFormItem :label="$t('page.manage.log.byUser')" path="byUser" class="w-65 pr-24px">
            <NSelect
              v-model:value="model.byUser"
              filterable
              clearable
              :options="memberOptions"
              :placeholder="$t('page.manage.log.form.byUser')"
            />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.logDetailType')" path="logDetailType">
            <NSelect
              v-model:value="model.logDetailType"
              :placeholder="$t('page.manage.log.form.logType')"
              :options="translateOptions(logDetailTypeOptions)"
              filterable
              clearable
            />
          </NFormItem>
        </div>

        <div v-show="model.logType === '1'">
          <NFormItem :label="$t('page.manage.log.requestPath')" path="requestPath">
            <NInput v-model:value="model.requestPath" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.requestParams')" path="requestParams">
            <NInput v-model:value="model.requestParams" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.responseData')" path="responseData">
            <NInput v-model:value="model.responseData" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.userAgent')" path="userAgent">
            <NInput v-model:value="model.userAgent" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.processTime')" path="processTime">
            <NInput v-model:value="model.processTime" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.ipAddress')" path="ipAddress">
            <NInput v-model:value="model.ipAddress" />
          </NFormItem>

          <NFormItem :label="$t('page.manage.log.responseCode')" path="responseCode">
            <NInput v-model:value="model.responseCode" :placeholder="$t('page.manage.log.form.responseCode')" />
          </NFormItem>
        </div>
      </NForm>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped></style>
