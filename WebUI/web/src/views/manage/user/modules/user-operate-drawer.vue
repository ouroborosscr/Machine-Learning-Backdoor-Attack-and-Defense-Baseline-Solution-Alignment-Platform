<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { fetchAddUser, fetchGetRoleList, fetchUpdateUser } from '@/service/api';
import { $t } from '@/locales';
import { statusTypeOptions, userGenderOptions } from '@/constants/business';

defineOptions({
  name: 'UserOperateDrawer'
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.User | null;
}

const props = defineProps<Props>();

interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: $t('page.manage.user.addUser'),
    edit: $t('page.manage.user.editUser')
  };
  return titles[props.operateType];
});

const model: Api.SystemManage.UserUpdateParams = reactive(createDefaultModel());

function createDefaultModel(): Api.SystemManage.UserUpdateParams {
  return {
    userName: '',
    password: '',
    userGender: '3',
    nickName: '',
    userPhone: '',
    userEmail: '',
    byUserRoleCodeList: ['R_USER'],
    statusType: '1'
  };
}

type RuleKey = Extract<keyof Api.SystemManage.UserUpdateParams, 'userName' | 'password' | 'nickName' | 'statusType'>;

const rules = ref<Record<RuleKey, App.Global.FormRule>>({
  userName: defaultRequiredRule,
  password: defaultRequiredRule,
  nickName: defaultRequiredRule,
  statusType: defaultRequiredRule
});

/** the enabled role options */
const roleOptions = ref<CommonType.Option<string>[]>([]);

async function getRoleOptions() {
  const { error, data } = await fetchGetRoleList({ size: 1000, statusType: '1' });

  if (!error) {
    const options = data.records.map(item => ({
      label: item.roleName,
      value: item.roleCode
    }));
    roleOptions.value = options;
  }
}

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    console.log('props.rowData', props.rowData);
    Object.assign(model, props.rowData);
  }

  if (props.operateType === 'add') {
    rules.value.password.required = true;
  } else if (props.operateType === 'edit') {
    rules.value.password.required = false;
  }
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  // request

  if (props.operateType === 'add') {
    const { error } = await fetchAddUser(model);
    if (!error) {
      window.$message?.success($t('common.addSuccess'));
    }
  } else if (props.operateType === 'edit') {
    console.log(model);
    const { error } = await fetchUpdateUser(model);
    if (!error) {
      window.$message?.success($t('common.updateSuccess'));
    }
  }

  closeDrawer();
  emit('submitted');
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
    getRoleOptions();
  }
});
</script>

<template>
  <NModal v-model:show="visible" preset="dialog" :title="title">
    <NForm
      ref="formRef"
      :model="model"
      :rules="rules"
      label-placement="left"
      label-width="auto"
      require-mark-placement="right-hanging"
      size="medium"
      class="max-w-150"
    >
      <NFormItem :label="$t('page.manage.user.userName')" path="userName">
        <NInput
          v-model:value="model.userName"
          :placeholder="$t('page.manage.user.form.userName')"
          :on-update-value="
            (value: string) => {
              model.nickName = value;
            }
          "
        />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.password')" path="password">
        <NInput v-model:value="model.password" :placeholder="$t('page.manage.user.form.password')" />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.userGender')" path="userGender">
        <NRadioGroup v-model:value="model.userGender">
          <NRadio v-for="item in userGenderOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
        </NRadioGroup>
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.nickName')" path="nickName">
        <NInput v-model:value="model.nickName" :placeholder="$t('page.manage.user.form.nickName')" />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.userPhone')" path="userPhone">
        <NInput v-model:value="model.userPhone" :placeholder="$t('page.manage.user.form.userPhone')" />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.userEmail')" path="email">
        <NInput v-model:value="model.userEmail" :placeholder="$t('page.manage.user.form.userEmail')" />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.userRole')" path="roles">
        <NSelect
          v-model:value="model.byUserRoleCodeList"
          multiple
          filterable
          clearable
          :options="roleOptions"
          :placeholder="$t('page.manage.user.form.userRole')"
        />
      </NFormItem>
      <NFormItem :label="$t('page.manage.user.userStatusType')" path="status">
        <NRadioGroup v-model:value="model.statusType">
          <NRadio v-for="item in statusTypeOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
        </NRadioGroup>
      </NFormItem>
    </NForm>
    <template #action>
      <NSpace :size="16">
        <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
        <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
