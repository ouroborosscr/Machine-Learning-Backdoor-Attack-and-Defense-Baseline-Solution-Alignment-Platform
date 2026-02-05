<template>
  <div class="h-full p-4 overflow-hidden">
    <n-card title="新建评估任务" class="h-full shadow-sm rounded-2xl">
      <div class="flex flex-col lg:flex-row gap-8 mt-4">
        <!-- 基础配置 -->
        <div class="flex-1 p-4 border border-gray-100 rounded-xl bg-gray-50/50 hover:shadow-sm transition-shadow flex flex-col gap-4">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="i-carbon-settings text-xl text-primary"></div>
              <span class="text-xl font-bold text-gray-700">基础配置</span>
            </div>
            <n-tag type="primary" size="small" :bordered="false" round>必选</n-tag>
          </div>

          <div class="flex flex-col gap-1">
            <span class="text-xs font-medium text-gray-500">数据集</span>
            <div class="flex gap-2">
              <n-select 
                v-model:value="form.basic.dataset" 
                :options="options.basic.dataset" 
                placeholder="选择数据集" 
                size="small"
                class="flex-1"
              />
              <n-button size="small" secondary @click="openUploadModal('dataset')">
                <template #icon><div class="i-carbon-upload" /></template>
                自定义
              </n-button>
            </div>
          </div>

          <div class="flex flex-col gap-1">
            <span class="text-xs font-medium text-gray-500">模型结构</span>
            <div class="flex gap-2">
              <n-select 
                v-model:value="form.basic.model" 
                :options="options.basic.model" 
                placeholder="选择模型结构" 
                size="small"
                class="flex-1"
              />
              <n-button size="small" secondary @click="openUploadModal('model')">
                <template #icon><div class="i-carbon-upload" /></template>
                自定义
              </n-button>
            </div>
          </div>

          <div class="flex flex-col gap-1">
            <span class="text-xs font-medium text-gray-500">模型权重 (可选)</span>
            <div class="flex gap-2">
              <n-select 
                v-model:value="form.basic.weights" 
                :options="options.basic.weights" 
                placeholder="选择权重" 
                size="small"
                class="flex-1"
              />
              <n-button size="small" secondary @click="openUploadModal('weights')">
                <template #icon><div class="i-carbon-upload" /></template>
                自定义
              </n-button>
            </div>
          </div>
          
          <div class="mt-1 text-xs text-gray-400">设置任务的基本运行环境与参数</div>
          
          <!-- 统一展示区域 -->
          <div class="mt-4 p-3 rounded-lg border border-gray-200 max-h-96 overflow-auto">
            <template v-for="item in basicSelectedItems" :key="item.value">
                <component
                  :is="getComponent(item.value)"
                  :title="item.label"
                />
              </template>
          </div>
        </div>

        <!-- 攻击评估 -->
        <div class="flex-1" :class="attackCardClass">
          <div class="flex items-center mb-4 h-7">
            <n-checkbox v-model:checked="form.attack.enabled" size="large">
              <span class="text-xl font-bold text-gray-700 ml-1">攻击评估</span>
            </n-checkbox>
          </div>

          <div class="flex flex-col gap-1">
            <n-checkbox v-model:checked="form.attack.adaptiveEnabled" :disabled="!form.attack.enabled">
              <span class="text-sm font-medium text-gray-600">后门投毒攻击</span>
            </n-checkbox>
            <n-select 
              v-model:value="form.attack.adaptive" 
              :options="options.attack.adaptive" 
              :disabled="!form.attack.enabled || !form.attack.adaptiveEnabled"
              placeholder="选择攻击方式 (可多选)" 
              multiple
              clearable
              size="small"
            />
          </div>

          <div class="flex flex-col gap-1">
            <n-checkbox v-model:checked="form.attack.naturalEnabled" :disabled="!form.attack.enabled">
              <span class="text-sm font-medium text-gray-600">自然后门攻击</span>
            </n-checkbox>
            <n-select 
              v-model:value="form.attack.natural" 
              :options="options.attack.natural" 
              :disabled="!form.attack.enabled || !form.attack.naturalEnabled"
              placeholder="选择攻击方式 (可多选)" 
              multiple
              clearable
              size="small"
            />
          </div>
          
          <div class="mt-2 text-xs text-gray-400">模拟多种后门攻击以测试模型鲁棒性</div>

          <!-- 统一展示区域，汇总所有选中的攻击方式 -->
          <div class="mt-4 p-3 rounded-lg border border-gray-200 max-h-96 overflow-auto">
            <template v-for="item in [...form.attack.adaptive, ...form.attack.natural]" :key="item">
              <component 
                :is="getComponent(item)" 
                :title="getLabel(item, 
                  options.attack.adaptive.find(o => o.value === item) ? options.attack.adaptive : options.attack.natural
                )"
              />
            </template>
          </div>
        </div>

        <!-- 后门检测 -->
        <div class="flex-1" :class="detectCardClass">
          <div class="flex items-center mb-4 h-7">
            <n-checkbox v-model:checked="form.detect.enabled" size="large">
              <span class="text-xl font-bold text-gray-700 ml-1">后门检测</span>
            </n-checkbox>
          </div>
          
          <div class="flex flex-col gap-3">
            <!-- 移除内联组件展示，只保留选择器 -->
            <div class="flex flex-col gap-1">
              <span class="text-xs font-bold text-gray-500">数据检测</span>
              <n-select 
                v-model:value="form.detect.dataTarget" 
                :options="options.detect.dataTarget" 
                :disabled="!form.detect.enabled"
                placeholder="选择检测目标" 
                size="small"
              />

              <div class="flex gap-2">
                <n-select 
                  v-model:value="form.detect.dataScheme" 
                  :options="options.detect.dataScheme" 
                  :disabled="!form.detect.enabled"
                  placeholder="选择检测方案" 
                  multiple
                  clearable
                  size="small"
                  class="flex-1"
                />
                <n-button 
                  size="small" 
                  secondary 
                  :disabled="!form.detect.enabled"
                  @click="openUploadModal('detectDataScheme')"
                >
                  <template #icon><div class="i-carbon-upload" /></template>
                  自定义
                </n-button>
              </div>
            </div>

            <!-- 移除内联组件展示，只保留选择器 -->
            <div class="flex flex-col gap-1">
              <span class="text-xs font-bold text-gray-500">模型检测</span>
              <n-select 
                v-model:value="form.detect.modelTarget" 
                :options="options.detect.modelTarget" 
                :disabled="!form.detect.enabled"
                placeholder="选择检测目标" 
                size="small"
              />

              <div class="flex gap-2">
                <n-select 
                  v-model:value="form.detect.modelScheme" 
                  :options="options.detect.modelScheme" 
                  :disabled="!form.detect.enabled"
                  placeholder="选择检测方案" 
                  multiple
                  clearable
                  size="small"
                  class="flex-1"
                />
                <n-button 
                  size="small" 
                  secondary 
                  :disabled="!form.detect.enabled"
                  @click="openUploadModal('detectModelScheme')"
                >
                  <template #icon><div class="i-carbon-upload" /></template>
                  自定义
                </n-button>
              </div>
            </div>
          </div>

          <div class="mt-2 text-xs text-gray-400">扫描系统中可能存在的恶意后门</div>

          <!-- 新增统一展示区域，汇总所有检测目标和方案 -->
          <div class="mt-4 p-3 rounded-lg border border-gray-200 max-h-96 overflow-auto">
            <template v-for="item in detectItems" :key="item.value">
              <component 
                :is="getComponent(item.value)" 
                :title="item.label"
              />
            </template>
          </div>
        </div>

        <!-- 后门修复 -->
        <div class="flex-1" :class="repairCardClass">
          <div class="flex items-center mb-4 h-7">
            <n-checkbox v-model:checked="form.repair.enabled" size="large">
              <span class="text-xl font-bold text-gray-700 ml-1">后门修复</span>
            </n-checkbox>
          </div>
          
          <div class="flex flex-col gap-3">
            <!-- 移除内联组件展示，只保留选择器 -->
            <div class="flex flex-col gap-1">
              <n-checkbox v-model:checked="form.repair.data.enabled" :disabled="!form.repair.enabled">
                <span class="text-sm font-medium text-gray-600">数据修复</span>
              </n-checkbox>
              <div class="flex gap-2">
                <n-select 
                  v-model:value="form.repair.data.schemes" 
                  :options="options.repair.data" 
                  :disabled="!form.repair.enabled || !form.repair.data.enabled"
                  placeholder="选择修复方案" 
                  multiple
                  clearable
                  size="small"
                  class="flex-1"
                />
                <n-button 
                  size="small" 
                  secondary 
                  :disabled="!form.repair.enabled || !form.repair.data.enabled"
                  @click="openUploadModal('repairData')"
                >
                  <template #icon><div class="i-carbon-upload" /></template>
                  自定义
                </n-button>
              </div>
            </div>

            <!-- 移除内联组件展示，只保留选择器 -->
            <div class="flex flex-col gap-1">
              <n-checkbox v-model:checked="form.repair.training.enabled" :disabled="!form.repair.enabled">
                <span class="text-sm font-medium text-gray-600">训练中缓解</span>
              </n-checkbox>
              <div class="flex gap-2">
                <n-select 
                  v-model:value="form.repair.training.schemes" 
                  :options="options.repair.training" 
                  :disabled="!form.repair.enabled || !form.repair.training.enabled"
                  placeholder="选择缓解方案" 
                  multiple
                  clearable
                  size="small"
                  class="flex-1"
                />
                <n-button 
                  size="small" 
                  secondary 
                  :disabled="!form.repair.enabled || !form.repair.training.enabled"
                  @click="openUploadModal('repairTraining')"
                >
                  <template #icon><div class="i-carbon-upload" /></template>
                  自定义
                </n-button>
              </div>
            </div>

            <!-- 移除内联组件展示，只保留选择器 -->
            <div class="flex flex-col gap-1">
              <n-checkbox v-model:checked="form.repair.model.enabled" :disabled="!form.repair.enabled">
                <span class="text-sm font-medium text-gray-600">模型修复</span>
              </n-checkbox>
              <div class="flex gap-2">
                <n-select 
                  v-model:value="form.repair.model.schemes" 
                  :options="options.repair.model" 
                  :disabled="!form.repair.enabled || !form.repair.model.enabled"
                  placeholder="选择修复方案" 
                  multiple
                  clearable
                  size="small"
                  class="flex-1"
                />
                <n-button 
                  size="small" 
                  secondary 
                  :disabled="!form.repair.enabled || !form.repair.model.enabled"
                  @click="openUploadModal('repairModel')"
                >
                  <template #icon><div class="i-carbon-upload" /></template>
                  自定义
                </n-button>
              </div>
            </div>
          </div>

          <div class="mt-2 text-xs text-gray-400">自动或手动修复发现的安全隐患</div>

          <!-- 新增统一展示区域，汇总所有修复方案 -->
          <div class="mt-4 p-3 rounded-lg border border-gray-200 max-h-96 overflow-auto">
            <template v-for="item in repairItems" :key="item.value">
              <component 
                :is="getComponent(item.value)" 
                :title="item.label"
              />
            </template>
          </div>
        </div>
      </div>

      <template #action>
        <div class="flex items-center justify-between pt-4 border-t border-gray-100">
          <div class="text-sm text-gray-500">
            已选模块：{{ selectedModulesCount }} / 4
          </div>
          <n-button 
            type="primary" 
            size="large" 
            class="w-40 font-bold shadow-md shadow-primary/30" 
            :loading="loading"
            @click="handleStart"
          >
            开始评估
          </n-button>
        </div>
      </template>
    </n-card>

    <n-modal v-model:show="showUploadModal">
      <n-card
        style="width: 600px"
        :title="`上传自定义${uploadTitle}`"
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <n-upload
          multiple
          directory-dnd
          action="https://www.mocky.io/v2/5e4bafc63100007100d8b70f"
          :max="5"
        >
          <n-upload-dragger>
            <div class="mb-3">
              <div class="i-carbon-cloud-upload text-5xl text-primary/50"></div>
            </div>
            <n-text style="font-size: 16px">
              点击或者拖动文件到该区域来上传
            </n-text>
            <n-p depth="3" style="margin: 8px 0 0 0">
              请上传符合格式要求的{{ uploadTitle }}文件
            </n-p>
          </n-upload-dragger>
        </n-upload>
      </n-card>
    </n-modal>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref, computed } from 'vue';
import { useMessage } from 'naive-ui';
import MNIST from './components/descriptions/MNIST.vue';
import GTSRB from './components/descriptions/GTSRB.vue';
import ResNet8 from './components/descriptions/ResNet8.vue';
import DefaultDescription from './components/descriptions/DefaultDescription.vue';
import PubFig from './components/descriptions/PubFig.vue';
import IMDB from './components/descriptions/IMDB.vue';
import CIFAR10 from './components/descriptions/CIFAR10.vue';
import CIFAR100 from './components/descriptions/CIFAR100.vue';
import SimpleNet from './components/descriptions/SimpleNet.vue';
import VGG16 from './components/descriptions/VGG16.vue';
import ResNet18 from './components/descriptions/ResNet18.vue';
import BERT from './components/descriptions/BERT.vue';
import Qwen7B from './components/descriptions/Qwen7B.vue';
import BadNets from './components/descriptions/BadNets.vue';
import InvisibleBackdoor from './components/descriptions/InvisibleBackdoor.vue';
import CleanLabelAttack from './components/descriptions/CleanLabelAttack.vue';
import DynamicBackdoorAttack from './components/descriptions/DynamicBackdoorAttack.vue';
import MultiBackdoorAttack from './components/descriptions/MultiBackdoorAttack.vue';
import WatermarkBackdoorAttack from './components/descriptions/WatermarkBackdoorAttack.vue';
import HCB from './components/descriptions/HCB.vue';
import CASSOCK from './components/descriptions/CASSOCK.vue';
import LoRA from './components/descriptions/LoRA.vue';
import UniversalTrigger from './components/descriptions/UniversalTrigger.vue';
import SpecificSampleTrigger from './components/descriptions/SpecificSampleTrigger.vue';
import PromptEvaluation from './components/descriptions/PromptEvaluation.vue';
import SVD from './components/descriptions/SVD.vue';
import DeepKNN from './components/descriptions/DeepKNN.vue';
import STRIP from './components/descriptions/STRIP.vue';
import ONION from './components/descriptions/ONION.vue';
import ParaFuzz from './components/descriptions/ParaFuzz.vue';
import Steps from './components/descriptions/Steps.vue';
import DMS_Steps from './components/descriptions/DMS_Steps.vue';
import NeuralCleanse from './components/descriptions/NeuralCleanse.vue';
import MESA from './components/descriptions/MESA.vue';
import 数据清洗 from './components/descriptions/数据清洗.vue';
import 数据增强 from './components/descriptions/数据增强.vue';
import 数据过滤 from './components/descriptions/数据过滤.vue';
import 梯度裁剪 from './components/descriptions/梯度裁剪.vue';
import 梯度加噪 from './components/descriptions/梯度加噪.vue';
import 样本对齐 from './components/descriptions/样本对齐.vue';
import 剪枝 from './components/descriptions/剪枝.vue';
import 微调 from './components/descriptions/微调.vue';
import MachineUnlearning from './components/descriptions/MachineUnlearning.vue';
import ABS from './components/descriptions/ABS.vue';
import FreeEagle from './components/descriptions/FreeEagle.vue';
import DeBackdoor from './components/descriptions/DeBackdoor.vue';
import RAP from './components/descriptions/RAP.vue';
import LinkBreaker from './components/descriptions/LinkBreaker.vue';
import MNTD from './components/descriptions/MNTD.vue';
import SSBAs from './components/descriptions/SSBAs.vue';

const loading = ref(false);
const message = useMessage();
const showUploadModal = ref(false);
const currentUploadField = ref('');

const form = reactive({
  basic: {
    dataset: null,
    model: null,
    weights: 'default'
  },
  attack: { 
    enabled: false, 
    adaptiveEnabled: false,
    adaptive: [], 
    naturalEnabled: false,
    natural: [] 
  },
  detect: { 
    enabled: false, 
    dataTarget: null,
    dataScheme: [],
    modelTarget: null,
    modelScheme: []
  },
  repair: { 
    enabled: false,
    data: { enabled: false, schemes: [] },
    training: { enabled: false, schemes: [] },
    model: { enabled: false, schemes: [] }
  }
});

const options = reactive({
  basic: {
    dataset: [
      { label: 'MNIST', value: 'MNIST' },
      { label: 'GTSRB', value: 'GTSRB' },
      { label: 'CIFAR10', value: 'CIFAR10' },
      { label: 'CIFAR100', value: 'CIFAR100' },
      { label: 'PubFig', value: 'PubFig' },
      { label: '自定义', value: 'custom' }
    ],
    model: [
      { label: 'Simple Net', value: 'Simple Net' },
      { label: 'VGG16', value: 'VGG16' },
      { label: 'ResNet8', value: 'ResNet8' },
      { label: 'ResNet18', value: 'ResNet18' },
      { label: '自定义', value: 'custom' }
    ],
    weights: [
      { label: '默认', value: 'default' },
      { label: '自定义', value: 'custom' }
    ]
  },
  attack: {
    adaptive: [
      { label: 'BadNets', value: 'BadNets' },
      { label: 'SSBAs', value: 'SSBAs' },
      { label: 'CASSOCK', value: 'CASSOCK' },
      { label: '不可视后门', value: 'Invisible' },
      { label: '干净标签攻击', value: 'Clean Label' },
      { label: '动态后门攻击', value: 'Dynamic' },
      { label: '多后门攻击', value: 'Multi-backdoor' },
      { label: '水印后门攻击', value: 'Watermark' },
      { label: 'HCB', value: 'HCB' },
      { label: '高效微调', value: 'Efficient Fine-tuning' }
    ],
    natural: [
      { label: '通用触发器', value: 'Universal Trigger' },
      { label: '特定样本触发器', value: 'Specific Sample Trigger' },
      { label: '提示词评估', value: 'Prompt Evaluation' }
    ]
  },
  detect: {
    dataTarget: [
      { label: '攻击评估数据集', value: 'attack_eval_dataset' },
      { label: '自定义数据集', value: 'custom_dataset' }
    ],
    dataScheme: [
      { label: 'SVD', value: 'SVD' },
      { label: 'STRIP', value: 'STRIP' },
      { label: '深度k最近邻', value: 'Deep-kNN' },
      { label: 'ONION', value: 'ONION' },
      { label: 'ParaFuzz', value: 'ParaFuzz' },
      { label: '自定义', value: 'custom' }
    ],
    modelTarget: [
      { label: '攻击评估模型', value: 'attack_eval_model' },
      { label: '自定义模型', value: 'custom_model' }
    ],
    modelScheme: [
      { label: 'Neural Cleanse', value: 'Neural Cleanse' },
      { label: 'DeBackdoor', value: 'DeBackdoor' },
      { label: 'Steps', value: 'Steps' },
      { label: 'DMS-Steps', value: 'DMS-Steps' },
      { label: 'MESA', value: 'MESA' },
      { label: 'ABS', value: 'ABS' },
      { label: 'FreeEagle', value: 'FreeEagle' },
      { label: 'RAP', value: 'RAP' },
      { label: 'LinkBreaker', value: 'LinkBreaker' },
      { label: 'MNTD', value: 'MNTD' },
      { label: '自定义', value: 'custom' }
    ]
  },
  repair: {
    data: [
      { label: '数据清洗', value: 'Data Cleaning' },
      { label: '数据增强', value: 'Data Augmentation' },
      { label: '数据过滤', value: 'Data Filtering' },
      { label: '自定义', value: 'custom' }
    ],
    training: [
      { label: '梯度加噪', value: 'Gradient Noise' },
      { label: '样本对齐', value: 'Sample Alignment' },
      { label: '梯度裁剪', value: 'Gradient Clipping' },
      { label: '自定义', value: 'custom' }
    ],
    model: [
      { label: '剪枝', value: 'Pruning' },
      { label: '微调', value: 'Fine-tuning' },
      { label: '机器遗忘', value: 'Machine Unlearning' },
      { label: '自定义', value: 'custom' }
    ]
  }
});

const componentMap: Record<string, any> = {
  'MNIST': MNIST,
  'GTSRB': GTSRB,
  'ResNet8': ResNet8,
  'DefaultDescription': DefaultDescription,
  'PubFig': PubFig,
  'IMDB': IMDB,
  'CIFAR10': CIFAR10,
  'CIFAR100': CIFAR100,
  'VGG16': VGG16,
  'ResNet18': ResNet18,
  'BERT': BERT,
  'Qwen7B': Qwen7B,
  'BadNets': BadNets,
  'Invisible': InvisibleBackdoor,
  'Clean Label': CleanLabelAttack,
  'Dynamic': DynamicBackdoorAttack,
  'Multi-backdoor': MultiBackdoorAttack,
  'Watermark': WatermarkBackdoorAttack,
  'HCB': HCB,
  'CASSOCK': CASSOCK,
  'Efficient Fine-tuning': LoRA,
  'Universal Trigger': UniversalTrigger,
  'Specific Sample Trigger': SpecificSampleTrigger,
  'Prompt Evaluation': PromptEvaluation,
  'Simple Net': SimpleNet,
  'SVD': SVD,
  'Deep-kNN': DeepKNN,
  'STRIP': STRIP,
  'ONION': ONION,
  'ParaFuzz': ParaFuzz,
  'Steps': Steps,
  'DMS-Steps': DMS_Steps,
  'Neural Cleanse': NeuralCleanse,
  'MESA': MESA,
  'Data Cleaning': 数据清洗,
  'Data Augmentation': 数据增强,
  'Data Filtering': 数据过滤,
  'Gradient Clipping': 梯度裁剪,
  'Gradient Noise': 梯度加噪,
  'Sample Alignment': 样本对齐,
  'Pruning': 剪枝,
  'Fine-tuning': 微调,
  'Machine Unlearning': MachineUnlearning,
  'ABS': ABS,
  'FreeEagle': FreeEagle,
  'DeBackdoor': DeBackdoor,
  'RAP': RAP,
  'LinkBreaker': LinkBreaker,
  'MNTD': MNTD,
  'SSBAs': SSBAs
};

function getComponent(value: string) {
  return componentMap[value] || DefaultDescription;
}

function getLabel(value: string, optionsList: { label: string, value: string }[]) {
  const option = optionsList.find(opt => opt.value === value);
  return option ? option.label : value;
}

const basicSelectedItems = computed(() => {
  const items: { value: string; label: string }[] = [];

  if (form.basic.dataset && form.basic.dataset !== 'custom') {
    items.push({
      value: form.basic.dataset,
      label: getLabel(form.basic.dataset, options.basic.dataset)
    });
  }

  if (form.basic.model && form.basic.model !== 'custom') {
    items.push({
      value: form.basic.model,
      label: getLabel(form.basic.model, options.basic.model)
    });
  }

  return items;
});

const detectItems = computed(() => {
  const items: { value: string, label: string }[] = [];
  
  if (form.detect.dataTarget) {
    items.push({
      value: form.detect.dataTarget,
      label: getLabel(form.detect.dataTarget, options.detect.dataTarget)
    });
  }
  
  if (form.detect.dataScheme && form.detect.dataScheme.length > 0) {
    form.detect.dataScheme.forEach(scheme => {
      items.push({
        value: scheme,
        label: getLabel(scheme, options.detect.dataScheme)
      });
    });
  }
  
  if (form.detect.modelTarget) {
    items.push({
      value: form.detect.modelTarget,
      label: getLabel(form.detect.modelTarget, options.detect.modelTarget)
    });
  }
  
  if (form.detect.modelScheme && form.detect.modelScheme.length > 0) {
    form.detect.modelScheme.forEach(scheme => {
      items.push({
        value: scheme,
        label: getLabel(scheme, options.detect.modelScheme)
      });
    });
  }
  
  return items;
});

const repairItems = computed(() => {
  const items: { value: string, label: string }[] = [];
  
  if (form.repair.data.schemes && form.repair.data.schemes.length > 0) {
    form.repair.data.schemes.forEach(scheme => {
      items.push({
        value: scheme,
        label: getLabel(scheme, options.repair.data)
      });
    });
  }
  
  if (form.repair.training.schemes && form.repair.training.schemes.length > 0) {
    form.repair.training.schemes.forEach(scheme => {
      items.push({
        value: scheme,
        label: getLabel(scheme, options.repair.training)
      });
    });
  }
  
  if (form.repair.model.schemes && form.repair.model.schemes.length > 0) {
    form.repair.model.schemes.forEach(scheme => {
      items.push({
        value: scheme,
        label: getLabel(scheme, options.repair.model)
      });
    });
  }
  
  return items;
});

const uploadTitle = computed(() => {
  const map: Record<string, string> = {
    dataset: '数据集',
    model: '模型结构',
    weights: '模型权重',
    detectDataScheme: '数据检测方案',
    detectModelScheme: '模型检测方案',
    repairData: '数据修复方案',
    repairTraining: '训练中缓解方案',
    repairModel: '模型修复方案'
  };
  return map[currentUploadField.value] || '文件';
});

const baseCardClass = 'p-4 border border-gray-100 rounded-xl transition-all duration-300 flex flex-col';
const activeCardClass = 'bg-white hover:shadow-sm';
const inactiveCardClass = 'bg-gray-50 opacity-60 grayscale';

const attackCardClass = computed(() => [
  baseCardClass,
  form.attack.enabled ? activeCardClass : inactiveCardClass
]);

const detectCardClass = computed(() => [
  baseCardClass,
  form.detect.enabled ? activeCardClass : inactiveCardClass
]);

const repairCardClass = computed(() => [
  baseCardClass,
  form.repair.enabled ? activeCardClass : inactiveCardClass
]);

function openUploadModal(field: string) {
  currentUploadField.value = field;
  showUploadModal.value = true;
}

const selectedModulesCount = computed(() => {
  let count = 1; 
  if (form.attack.enabled) count++;
  if (form.detect.enabled) count++;
  if (form.repair.enabled) count++;
  return count;
});

async function handleStart() {
  // 1. 基础校验
  if (!form.basic.dataset || !form.basic.model) {
    message.warning('请完善基础配置（数据集和模型结构为必填）');
    return;
  }

  // 开启加载状态
  loading.value = true;

  try {
    // 2. 准备日志数据 (根据表单和真实时间)
    const now = new Date();
    // 格式化时间: YYYY/MM/DD/HH/mm
    const timeStr = `${now.getFullYear()}/${(now.getMonth() + 1).toString().padStart(2, '0')}/${now.getDate().toString().padStart(2, '0')}/${now.getHours().toString().padStart(2, '0')}/${now.getMinutes().toString().padStart(2, '0')}`;
    
    // 生成ID: 'a' + 时间戳 + 3位随机数
    const uniqueId = `a${Date.now()}${Math.floor(Math.random() * 1000)}`;

    // 收集所有选中的评估项目名称
    const selectedItems: string[] = [];
    if (form.attack.enabled) {
      selectedItems.push(...form.attack.adaptive, ...form.attack.natural);
    }
    if (form.detect.enabled) {
      if (form.detect.dataTarget) selectedItems.push(form.detect.dataTarget); // 包含目标本身
      selectedItems.push(...form.detect.dataScheme);
      if (form.detect.modelTarget) selectedItems.push(form.detect.modelTarget);
      selectedItems.push(...form.detect.modelScheme);
    }
    if (form.repair.enabled) {
      selectedItems.push(
        ...form.repair.data.schemes,
        ...form.repair.training.schemes,
        ...form.repair.model.schemes
      );
    }
    // 过滤掉空值或undefined
    const cleanItems = selectedItems.filter(Boolean);

    // 构造日志 JSON 对象
    const logData = {
      "id": uniqueId,
      "数据集": form.basic.dataset,
      "模型结构": form.basic.model,
      "开始时间": timeStr,
      "完成时间": timeStr, // 根据需求，此处暂时设为与开始时间相同
      "评估项目": cleanItems,
      "下载报告": `http://127.0.0.1:9999/api/v1/system-manage/file-download/download?name=MNIST%E8%AF%84%E4%BC%B0%E6%8A%A5%E5%91%8A.pdf`,
      "应用测试": `http://10.161.10.20:8003/?data=MNIST`,
      "导出API": `./api/${uniqueId}`,
      "导出安全模型": `./model/${uniqueId}.pt`,
      "导出安全数据": `./dataset/${uniqueId}.zip`
    };

    // 3. 调用 API
    // 注意：JSON字符串需要编码以放入URL中
    const jsonParam = encodeURIComponent(JSON.stringify(logData));
    await fetch(`http://127.0.0.1:9999/api/v1/system-manage/edit-logs?json_data=${jsonParam}`);

    // === 下面是原有的跳转逻辑 ===

    const evaluationConfig: any = {
      "攻击评估": {},
      "后门检测与修复": {
        "数据处理阶段": {},
        "模型训练阶段": {},
        "模型评估阶段": {}
      },
      "修复后评估": {}
    };

    // --- 攻击评估部分 ---
    if (form.attack.adaptiveEnabled && form.attack.adaptive.length > 0) {
      form.attack.adaptive.forEach((attack: string) => {
        if (attack === "BadNets") {
          evaluationConfig["攻击评估"]["BadNets数据投毒"] = "攻击模拟";
          evaluationConfig["修复后评估"]["BadNets效果评估"] = "攻击模拟";
        }
        if (attack === "SSBAs") {
          evaluationConfig["攻击评估"]["SSBAs数据投毒"] = "攻击模拟";
          evaluationConfig["修复后评估"]["SSBAs效果评估"] = "攻击模拟";
        }
        if (attack === "CASSOCK") {
          evaluationConfig["攻击评估"]["CASSOCK攻击训练"] = "攻击模拟";
          evaluationConfig["修复后评估"]["CASSOCK效果评估"] = "攻击模拟";
        }
      });
    }

    // --- natural 自然后门攻击 ---
    if (form.attack.naturalEnabled && form.attack.natural.length > 0) {
      form.attack.natural.forEach((attack: string) => {
        if (attack === "Universal Trigger") {
          evaluationConfig["攻击评估"]["通用触发器生成"] = "攻击模拟";
          evaluationConfig["修复后评估"]["通用触发器效果评估"] = "攻击模拟";
        }
        if (attack === "Specific Sample Trigger") {
          evaluationConfig["攻击评估"]["样本专用触发器生成"] = "攻击模拟";
          evaluationConfig["修复后评估"]["样本专用触发器效果评估"] = "攻击模拟";
        }
      });
    }

    // --- 后门检测部分 ---
    // ---- 数据处理阶段 ----
    if (form.detect.dataScheme) {
      form.detect.dataScheme.forEach((scheme: string) => {
        if (scheme === "SVD" || scheme === "STRIP") {
          evaluationConfig["后门检测与修复"]["数据处理阶段"][scheme] = "后门检测与修复";
        }
      });
    }

    // ---- 模型评估阶段（检测）----
    if (form.detect.modelScheme) {
      form.detect.modelScheme.forEach((scheme: string) => {
        if (["Neural Cleanse", "DeBackdoor", "Steps"].includes(scheme)) {
          evaluationConfig["后门检测与修复"]["模型评估阶段"][scheme] = "后门检测与修复";
        }
      });
    }

    // --- 后门修复部分 ---
    // ---- 数据处理阶段（修复）----
    if (form.repair.data.schemes) {
      form.repair.data.schemes.forEach((scheme: string) => {
        if (scheme === "Data Cleaning") {
          evaluationConfig["后门检测与修复"]["数据处理阶段"]["数据清洗"] = "后门修复";
        }
        if (scheme === "Data Augmentation") {
          evaluationConfig["后门检测与修复"]["数据处理阶段"]["数据增强"] = "后门修复";
        }
      });
    }

    // ---- 模型训练阶段 ----
    if (form.repair.training.schemes) {
      form.repair.training.schemes.forEach((scheme: string) => {
        if (scheme === "Gradient Noise") {
          evaluationConfig["后门检测与修复"]["模型训练阶段"]["梯度加噪"] = "后门修复";
        }
        if (scheme === "Sample Alignment") {
          evaluationConfig["后门检测与修复"]["模型训练阶段"]["样本对齐"] = "后门修复";
        }
      });
    }

    // ---- 模型评估阶段（修复）----
    if (form.repair.model.schemes) {
      form.repair.model.schemes.forEach((scheme: string) => {
        if (scheme === "Pruning") {
          evaluationConfig["后门检测与修复"]["模型评估阶段"]["剪枝"] = "后门修复";
        }
        if (scheme === "Fine-tuning") {
          evaluationConfig["后门检测与修复"]["模型评估阶段"]["微调"] = "后门修复";
        }
        if (scheme === "Machine Unlearning") {
          evaluationConfig["后门检测与修复"]["模型评估阶段"]["机器遗忘"] = "后门修复";
        }
      });
    }

    // 将JSON对象转换为字符串并进行URL编码
    const configJsonString = encodeURIComponent(JSON.stringify(evaluationConfig));
    
    // 获取数据集名称并进行URL编码
    const datasetName = encodeURIComponent(form.basic.dataset);
    
    // 跳转到/process页面并传递两个参数：config和dataset
    window.location.href = `/process?config=${configJsonString}&dataset=${datasetName}`;

  } catch (error) {
    console.error("操作失败:", error);
    message.error("创建任务失败，请检查网络或配置");
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
</style>
