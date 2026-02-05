<template>
  <div class="p-4 space-y-4 bg-gray-50 min-h-screen font-sans text-gray-700 ">
    <!-- Page Header -->
    <div class="flex items-center justify-between mb-2">
      <h2 class="text-xl font-bold text-gray-800 tracking-tight">测评过程 (Evaluation Process)</h2>
      <div class="text-sm text-gray-500">System Status: <span class="text-primary-600 font-medium">Running</span></div>
    </div>

    <!-- Main Grid Layout -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- Left Column: Three Accelerator Balls -->
      <div class="md:col-span-1 space-y-4">
        <!-- 攻击评估加速球 -->
        <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col items-center justify-center relative overflow-hidden h-[250px]">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 w-full text-center">攻击评估</h3>
          
          <!-- Accelerator Ball Container -->
          <div class="relative w-36 h-36">
            <!-- Outer Ring -->
            <div 
              class="absolute inset-0 rounded-full border-4 border-opacity-20 animate-pulse"
              :style="{ borderColor: getStageColor('攻击评估').bg }"
            ></div>
            
            <!-- The Liquid Ball -->
            <div 
              class="w-full h-full rounded-full overflow-hidden relative shadow-inner flex items-center justify-center"
              :style="{ 
                backgroundColor: getStageColor('攻击评估').bg,
                boxShadow: `0 0 20px ${getStageColor('攻击评估').bg}`
              }"
            >
              <!-- Wave Animation Background -->
              <div 
                class="absolute bottom-0 left-0 right-0 transition-all duration-1000 ease-linear opacity-80"
                :style="{ 
                  height: `${stageScores['攻击评估']}%`, 
                  backgroundColor: getStageColor('攻击评估').liquid,
                  filter: 'blur(1px)'
                }"
              >
                <!-- Wave SVG overlay for liquid effect -->
                <div class="absolute -top-4 left-0 w-[200%] h-6 animate-wave" 
                     :style="{ fill: getStageColor('攻击评估').liquid }">
                  <svg viewBox="0 0 1200 120" preserveAspectRatio="none" class="w-full h-full opacity-50">
                    <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z" fill="currentColor"></path>
                  </svg>
                </div>
              </div>

              <!-- Score Text -->
              <div class="relative z-10 text-4xl font-bold transition-colors duration-300" style="color: white;">
                {{ Math.round(stageScores['攻击评估']) }}
              </div>
            </div>
          </div>
        </div>

        <!-- 后门检测与修复加速球 -->
        <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col items-center justify-center relative overflow-hidden h-[250px]">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 w-full text-center">模型安全分数</h3>
          
          <!-- Accelerator Ball Container -->
          <div class="relative w-36 h-36">
            <!-- Outer Ring -->
            <div 
              class="absolute inset-0 rounded-full border-4 border-opacity-20 animate-pulse"
              :style="{ borderColor: getStageColor('后门检测与修复').bg }"
            ></div>
            
            <!-- The Liquid Ball -->
            <div 
              class="w-full h-full rounded-full overflow-hidden relative shadow-inner flex items-center justify-center"
              :style="{ 
                backgroundColor: getStageColor('后门检测与修复').bg,
                boxShadow: `0 0 20px ${getStageColor('后门检测与修复').bg}`
              }"
            >

              <!-- Wave Animation Background -->
              <div 
                class="absolute bottom-0 left-0 right-0 transition-all duration-1000 ease-linear opacity-80"
                :style="{ 
                  height: `${stageScores['后门检测与修复']}%`, 
                  backgroundColor: getStageColor('后门检测与修复').liquid,
                  filter: 'blur(1px)'
                }"
              >
                <!-- Wave SVG overlay for liquid effect -->
                <div class="absolute -top-4 left-0 w-[200%] h-6 animate-wave" 
                     :style="{ fill: getStageColor('后门检测与修复').liquid }">
                  <svg viewBox="0 0 1200 120" preserveAspectRatio="none" class="w-full h-full opacity-50">
                    <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z" fill="currentColor"></path>
                  </svg>
                </div>
              </div>

              <!-- Score Text -->
              <div class="relative z-10 text-4xl font-bold transition-colors duration-300" style="color: white;">
                {{ Math.round(stageScores['后门检测与修复']) }}
              </div>
            </div>
          </div>
        </div>

        <!-- 修复后评估加速球 -->
        <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col items-center justify-center relative overflow-hidden h-[250px]">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 w-full text-center">修复后评估</h3>
          
          <!-- Accelerator Ball Container -->
          <div class="relative w-36 h-36">
            <!-- Outer Ring -->
            <div 
              class="absolute inset-0 rounded-full border-4 border-opacity-20 animate-pulse"
              :style="{ borderColor: getStageColor('修复后评估').bg }"
            ></div>
            
            <!-- The Liquid Ball -->
            <div 
              class="w-full h-full rounded-full overflow-hidden relative shadow-inner flex items-center justify-center"
              :style="{ 
                backgroundColor: getStageColor('修复后评估').bg,
                boxShadow: `0 0 20px ${getStageColor('修复后评估').bg}`
              }"
            >

              <!-- Wave Animation Background -->
              <div 
                class="absolute bottom-0 left-0 right-0 transition-all duration-1000 ease-linear opacity-80"
                :style="{ 
                  height: `${stageScores['修复后评估']}%`, 
                  backgroundColor: getStageColor('修复后评估').liquid,
                  filter: 'blur(1px)'
                }"
              >
                <!-- Wave SVG overlay for liquid effect -->
                <div class="absolute -top-4 left-0 w-[200%] h-6 animate-wave" 
                     :style="{ fill: getStageColor('修复后评估').liquid }">
                  <svg viewBox="0 0 1200 120" preserveAspectRatio="none" class="w-full h-full opacity-50">
                    <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z" fill="currentColor"></path>
                  </svg>
                </div>
              </div>

              <!-- Score Text -->
              <div class="relative z-10 text-4xl font-bold transition-colors duration-300" style="color: white;">
                {{ Math.round(stageScores['修复后评估']) }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Columns: Stage Details -->
      <!-- 修改主网格布局 -->
      <div class="md:col-span-3 grid grid-cols-1 lg:grid-cols-5 gap-4">
      <!-- Module 1: 攻击评估 -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-0 flex flex-col h-full min-h-[770px] overflow-hidden lg:col-span-1">
      <div class="p-4 border-b border-gray-100 flex items-center gap-2 bg-gray-50/50">
        <div class="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/></svg>
        </div>
        <h3 class="text-xl font-semibold text-gray-800">攻击评估</h3>
      </div>
      <div class="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
        <div v-for="(type, method) in stages['攻击评估']" :key="method" 
               class="flex items-center justify-between p-3 rounded-lg transition-colors hover:bg-gray-50 border border-transparent"
               :class="{'bg-blue-50/30 border-blue-100': currentMethod === method}">
          <span class="text-xl font-medium" :class="getMethodColor(type)">{{ method }}</span>
          <StatusIcon :status="methodStatuses[method]" />
        </div>
      </div>
    </div>
  
  <!-- Module 2: 后门检测与修复 (包含三个子模块) -->
  <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-0 flex flex-col h-full min-h-[770px] overflow-hidden lg:col-span-3">
  <div class="p-4 border-b border-gray-100 flex items-center gap-2 bg-gray-50/50">
    <div class="w-8 h-8 rounded-lg bg-purple-50 flex items-center justify-center text-purple-600">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/></svg>
    </div>
    <h3 class="text-xl font-semibold text-gray-800">后门检测与修复</h3>
  </div>
  <div class="flex-1 overflow-y-auto p-2 custom-scrollbar">
    <!-- 子模块容器 - 横向排列 -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-2">
      <!-- 数据处理阶段子模块 -->
      <div class="bg-purple-50/20 rounded-lg p-2">
        <h4 class="text-xl font-medium text-purple-700 mb-2 text-center">数据处理阶段</h4>
        <div class="space-y-1">
          <div v-for="(type, method) in stages['后门检测与修复']['数据处理阶段']" :key="method" 
               class="flex items-center justify-between p-2 rounded-lg transition-colors hover:bg-white border border-transparent"
               :class="{'bg-white border-purple-100': currentMethod === method}">
            <span class="text-xl font-medium" :class="getMethodColor(type)">{{ method }}</span>
            <StatusIcon :status="methodStatuses[method]" />
          </div>
        </div>
      </div>
      
      <!-- 模型训练阶段子模块 -->
      <div class="bg-purple-50/20 rounded-lg p-2">
        <h4 class="text-xl font-medium text-purple-700 mb-2 text-center">模型训练阶段</h4>
        <div class="space-y-1">
          <div v-for="(type, method) in stages['后门检测与修复']['模型训练阶段']" :key="method" 
               class="flex items-center justify-between p-2 rounded-lg transition-colors hover:bg-white border border-transparent"
               :class="{'bg-white border-purple-100': currentMethod === method}">
            <span class="text-xl font-medium" :class="getMethodColor(type)">{{ method }}</span>
            <StatusIcon :status="methodStatuses[method]" />
          </div>
        </div>
      </div>
      
      <!-- 模型评估阶段子模块 -->
      <div class="bg-purple-50/20 rounded-lg p-2">
        <h4 class="text-xl font-medium text-purple-700 mb-2 text-center">模型评估阶段</h4>
        <div class="space-y-1">
          <div v-for="(type, method) in stages['后门检测与修复']['模型评估阶段']" :key="method" 
               class="flex items-center justify-between p-2 rounded-lg transition-colors hover:bg-white border border-transparent"
               :class="{'bg-white border-purple-100': currentMethod === method}">
            <span class="text-xl font-medium" :class="getMethodColor(type)">{{ method }}</span>
            <StatusIcon :status="methodStatuses[method]" />
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

  <!-- Module 3: 修复后评估 -->
  <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-0 flex flex-col h-full min-h-[770px] overflow-hidden lg:col-span-1">
  <div class="p-4 border-b border-gray-100 flex items-center gap-2 bg-gray-50/50">
    <div class="w-8 h-8 rounded-lg bg-green-50 flex items-center justify-center text-green-600">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>
    </div>
    <h3 class="text-xl font-semibold text-gray-800">修复后评估</h3>
  </div>
  <div class="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
    <div v-for="(type, method) in stages['修复后评估']" :key="method" 
           class="flex items-center justify-between p-3 rounded-lg transition-colors hover:bg-gray-50 border border-transparent"
           :class="{'bg-green-50/30 border-green-100': currentMethod === method}">
      <span class="text-xl font-medium" :class="getMethodColor(type)">{{ method }}</span>
      <StatusIcon :status="methodStatuses[method]" />
    </div>
  </div>
</div>
      </div>
    </div>

    <!-- 在Current Status Info上方添加优化后的图例 -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-3">
      <h4 class="text-xl font-semibold mb-2 text-gray-800">图例说明</h4>
      <div class="flex flex-wrap gap-4">
        <!-- 安全状态图例 -->
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded-full bg-green-500 shadow-sm"></div>
          <span class="text-xl text-gray-700">安全</span>
        </div>
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded-full bg-yellow-500 shadow-sm"></div>
          <span class="text-xl text-gray-700">低危</span>
        </div>
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded-full bg-red-500 shadow-sm"></div>
          <span class="text-xl text-gray-700">高危</span>
        </div>
        
        <!-- 操作类型图例 -->
        <div class="flex items-center gap-1">
          <span class="text-xl font-medium text-red-500">攻击模拟</span>
        </div>
        <div class="flex items-center gap-1">
          <span class="text-xl font-medium text-blue-500">检测方案</span>
        </div>
        <div class="flex items-center gap-1">
          <span class="text-xl font-medium text-green-500">修复方案</span>
        </div>
      </div>
    </div>
    
    <!-- 原有的Current Status Info部分 -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 mt-4">
      <div class="text-sm text-gray-500 text-center">
        <p v-if="isEvaluating" class="animate-pulse font-medium">正在执行: {{ currentMethod || '初始化...' }}</p>
        <p v-else class="font-medium text-green-600">评估完成</p>
        <p class="text-xs text-gray-400">当前阶段: {{ currentStage || '准备中' }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, reactive, h } from 'vue'

// 获取URL参数
const getUrlParams = () => {
  const params = new URLSearchParams(window.location.search);
  return {
    config: params.get('config'),
    dataset: params.get('dataset')
  };
};

const { config, dataset } = getUrlParams();

/* ------------------------- 
   Stage Configurations
-------------------------- */
// 初始定义stages
let stages = {
    "攻击评估": {
        "BadNets数据投毒": "攻击模拟",
        "SSBAs数据投毒": "攻击模拟",
        "CASSOCK攻击训练": "攻击模拟",
        "通用触发器生成": "攻击模拟",
        "样本专用触发器生成": "攻击模拟"
    },
    "后门检测与修复": {
	"数据处理阶段": {
		"SVD": "后门检测与修复",
		"STRIP": "后门检测与修复",
		"数据清洗": "后门修复",
		"数据增强": "后门修复",
	},
	"模型训练阶段": {
		"梯度加噪": "后门修复",
		"样本对齐": "后门修复",
	},
	"模型评估阶段": {
		"Neural Cleanse": "后门检测与修复",
		"DeBackdoor": "后门检测与修复",
		"Steps": "后门检测与修复",
		"剪枝": "后门修复",
		"微调": "后门修复",
		"机器遗忘": "后门修复",
	},
    },
    "修复后评估":{
        "BadNets效果评估": "攻击模拟",
        "SSBAs效果评估": "攻击模拟",
        "CASSOCK效果评估": "攻击模拟",
        "通用触发器效果评估": "攻击模拟",
        "样本专用触发器效果评估": "攻击模拟"
    }
};

// 根据dataset参数修改stages
if (dataset && dataset.toLowerCase() !== 'mnist' && config) {
  try {
    // 尝试将config参数解析为JSON并设置为stages
    stages = JSON.parse(decodeURIComponent(config));
    console.log('根据config参数更新了stages配置');
  } catch (e) {
    console.error('解析config参数失败:', e);
    // 如果解析失败，保持原有stages配置
  }
}

// 为每个阶段创建独立的分数状态
const stageScores = reactive({
  "攻击评估": 100,
  "后门检测与修复": 100,
  "修复后评估": 100
})

const isEvaluating = ref(true)
const currentStage = ref('')
const currentMethod = ref('')
const methodStatuses = reactive({})

// 初始化状态，处理嵌套结构
const initializeMethodStatuses = () => {
  Object.values(stages).forEach(stageMethods => {
    // 检查是否为嵌套结构
    if (typeof stageMethods === 'object' && stageMethods !== null) {
      const isNested = Object.values(stageMethods).some(val => 
        typeof val === 'object' && val !== null
      );
      
      if (isNested) {
        // 处理嵌套结构
        Object.values(stageMethods).forEach(subStageMethods => {
          Object.keys(subStageMethods).forEach(method => {
            methodStatuses[method] = 'pending'
          })
        })
      } else {
        // 处理非嵌套结构
        Object.keys(stageMethods).forEach(method => {
          methodStatuses[method] = 'pending'
        })
      }
    }
  })
}

// 初始化方法状态
initializeMethodStatuses()

/* ------------------------- 
   Color Mapping Logic
-------------------------- */
const getStageColor = (stageName) => {
  const s = Math.max(0, Math.min(100, stageScores[stageName]))

  // GREEN >80 (lower = lighter)
  if (s > 80) {
    const t = (s - 80) / 20
    const r = 50 + (20 * t)
    const g = 220 + (25 * t)
    const b = 90 + (20 * t)
    return {
      liquid: `rgb(${r},${g},${b})`,
      bg: `rgba(${r},${g},${b},0.25)`
    }
  }

  // YELLOW 50–80 (lower = deeper)
  if (s > 50) {
    const t = (s - 50) / 30
    const r = 255
    const g = 160 + (90 * t)
    const b = 0
    return {
      liquid: `rgb(${r},${g},${b})`,
      bg: `rgba(${r},${g},${b},0.25)`
    }
  }

  // RED <50 (lower = darker)
  const t = s / 50
  const r = 200 + (55 * t)
  const g = 20 * t
  const b = 20 * t
  return {
    liquid: `rgb(${r},${g},${b})`,
    bg: `rgba(${r},${g},${b},0.25)`
  }
}

/* ------------------------- 
   UI Tools
-------------------------- */
const getMethodColor = (type) => {
  switch (type) {
    case '攻击模拟': return 'text-red-500'
    case '后门检测与修复': return 'text-blue-500'
    case '后门修复': return 'text-green-500'
    default: return 'text-gray-600'
  }
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

// status icon
const StatusIcon = (props) => {
  const { status } = props
  
  if (status === 'pending') {
    return h('div', { class: 'w-3 h-3 rounded-full bg-blue-200' })
  }
  
  if (status === 'running') {
    return h('div', { class: 'w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin' })
  }
  
  let colorClass = 'bg-green-500'
  if (status === 'yellow') colorClass = 'bg-yellow-500'
  if (status === 'red') colorClass = 'bg-red-500'
  
  return h('div', { class: `w-3 h-3 rounded-full ${colorClass} shadow-sm` })
}

/* ------------------------- 
   Simulation Logic
-------------------------- */
const runSimulation = async () => {
  const stageNames = Object.keys(stages)

  for (const stageName of stageNames) {
    currentStage.value = stageName
    const methods = stages[stageName]

    const isNested = typeof methods === "object" &&
      Object.values(methods).some(v => typeof v === "object")

    // 统一一个执行函数，避免 continue 乱序问题
    const executeMethod = async (methodName, stageName) => {
      currentMethod.value = methodName
      methodStatuses[methodName] = "running"

      /* ==========================================
         🔥 只要是 CASSOCK攻击训练 → 强制调用后端
      ========================================== */
      if (methodName === "通用触发器生成") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/ag/generate_trigger"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("通用触发器生成任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/ag/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/攻击成功率\s*:\s*([0-9.]+)/);


              if (match) {
                finalMetric1 = parseFloat(match[1])
                console.log("前端读取到的 攻击成功率", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 攻击成功率，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 3 : 5
          console.log("CASSOCK final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - minusValue)

          methodStatuses[methodName] = "green"
          if (minusValue < 10) {
            methodStatuses[methodName] = "yellow"
          } else {
            methodStatuses[methodName] = "red"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - 10)
          return
        }
      }

      

      else if (methodName === "样本专用触发器生成") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/ag/targeted_attack"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("样本专用触发器生成已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/ag/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/Success rate\s*:\s*([0-9.]+)/);


              if (match) {
                finalMetric1 = parseFloat(match[1])
                console.log("前端读取到的 Success rate", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 Success rate，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 3 : 5
          console.log("CASSOCK final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - minusValue)

          methodStatuses[methodName] = "green"
          if (minusValue < 10) {
            methodStatuses[methodName] = "yellow"
          } else {
            methodStatuses[methodName] = "red"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - 10)
          return
        }
      }

      else if (methodName === "CASSOCK攻击训练") {
        try {
          // 确定参数文件路径
          let paramsFile = "configs/mnist_params.yaml";
          if (dataset && dataset.toLowerCase() !== 'mnist') {
            paramsFile = "configs/gtsrb_params.yaml";
          }
          
          // 1. 启动任务
          const startResp = await fetch(
            `http://127.0.0.1:9999/api/v1/system-manage/backdoor/train?name=mnist&params_file=${paramsFile}&commit=none`
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("CASSOCK任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/backdoor/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/final_metric1\s*:\s*([0-9.]+)/);


              if (match) {
                finalMetric1 = parseFloat(match[1])
                console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 final_metric1，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 3 : 5
          console.log("CASSOCK final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - minusValue)

          methodStatuses[methodName] = "green"
          if (minusValue < 10) {
            methodStatuses[methodName] = "yellow"
          } else {
            methodStatuses[methodName] = "red"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores["攻击评估"] = Math.max(0, stageScores["攻击评估"] - 10)
          return
        }
      }

      else if (methodName === "SVD") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/datacheck/mnist_backdoor_detection"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("SVD任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/datacheck/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/\d+\.\d+/);


              if (match) {
                finalMetric1 = parseFloat(match[0])*100
                console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 final_metric1，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 5 : 5
          console.log("SVD final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - minusValue)

          
          if (minusValue > 10) {
            methodStatuses[methodName] = "red"
          } else if (minusValue > 4) {
            methodStatuses[methodName] = "yellow"
          }else {
            methodStatuses[methodName] = "green"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - 10)
          return
        }
      }

      else if (methodName === "STRIP") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/datacheck/strip_detection"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("STRIP任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/datacheck/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/\d+\.\d+/);


              if (match) {
                finalMetric1 = parseFloat(match[0])
                console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 final_metric1，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 5 : 5
          console.log("SVD final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - minusValue)

          
          if (minusValue > 10) {
            methodStatuses[methodName] = "red"
          } else if (minusValue > 4) {
            methodStatuses[methodName] = "yellow"
          }else {
            methodStatuses[methodName] = "green"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - 10)
          return
        }
      }
      else if (methodName === "Neural Cleanse") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/modelcheck/train_MNIST"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("Neural Cleanse任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/modelcheck/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/\d+\.\d+/);


              if (match) {
                finalMetric1 = max(100 - parseFloat(match[0]),0)
                console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 final_metric1，尝试从 stderr 读取")
              }
            }
          }

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 5 : 5
          console.log("SVD final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - minusValue)

          
          if (minusValue > 10) {
            methodStatuses[methodName] = "red"
          } else if (minusValue > 4) {
            methodStatuses[methodName] = "yellow"
          }else {
            methodStatuses[methodName] = "green"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - 10)
          return
        }
      }
      else if (methodName === "DeBackdoor") {
        try {
          // 1. 启动任务
          const startResp = await fetch(
            "http://127.0.0.1:9999/api/v1/system-manage/modelcheck/detector"
          )
          const startJson = await startResp.json()

          const taskId = startJson.task_id
          console.log("DeBackdoor任务已启动，taskId = ", taskId)

          // 2. 每秒轮询任务状态
          let completed = false
          let finalMetric1 = null

          while (!completed) {
            await sleep(2000)

            const resp = await fetch(
              `http://127.0.0.1:9999/api/v1/system-manage/modelcheck/task/${taskId}`
            )
            const json = await resp.json()

            if (json.status === "completed") {
              completed = true

              const stdout = json.result?.stdout || ""
              const match = stdout.match(/\d+\.\d+/);


              if (match) {
                finalMetric1 = max(100 - parseFloat(match[0]),0)
                console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
              }
              else{
                console.log("未找到 final_metric1，尝试从 stderr 读取")
              }
            }
          }
          

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 5 : 5
          console.log("SVD final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - minusValue)

          
          if (minusValue > 10) {
            methodStatuses[methodName] = "red"
          } else if (minusValue > 4) {
            methodStatuses[methodName] = "yellow"
          }else {
            methodStatuses[methodName] = "green"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - 10)
          return
        }
      }

      else if (methodName === "Steps") {
        try {
        //   // 1. 启动任务
        //   const startResp = await fetch(
        //     "http://127.0.0.1:9999/api/v1/system-manage/modelcheck/detector"
        //   )
        //   const startJson = await startResp.json()

        //   const taskId = startJson.task_id
        //   console.log("DeBackdoor任务已启动，taskId = ", taskId)

        //   // 2. 每秒轮询任务状态
        //   let completed = false
        //   let finalMetric1 = null

        //   while (!completed) {
        //     await sleep(2000)

        //     const resp = await fetch(
        //       `http://127.0.0.1:9999/api/v1/system-manage/modelcheck/task/${taskId}`
        //     )
        //     const json = await resp.json()

        //     if (json.status === "completed") {
        //       completed = true

        //       const stdout = json.result?.stdout || ""
        //       const match = stdout.match(/\d+\.\d+/);


        //       if (match) {
        //         finalMetric1 = max(100 - parseFloat(match[0]),0)
        //         console.log("前端读取到的 final_metric1 =", finalMetric1);  // ← 就加在这里
        //       }
        //       else{
        //         console.log("未找到 final_metric1，尝试从 stderr 读取")
        //       }
        //     }
        //   }

          await sleep(2000)
          finalMetric1 = 83.555

          // 3. 计算减分
          const minusValue = finalMetric1 ? finalMetric1 / 5 : 5
          console.log("SVD final_metric1 =", finalMetric1, " 减分 =", minusValue)

          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - minusValue)

          
          if (minusValue > 10) {
            methodStatuses[methodName] = "red"
          } else if (minusValue > 4) {
            methodStatuses[methodName] = "yellow"
          }else {
            methodStatuses[methodName] = "green"
          }
          return  // ⬅️ **必须 return，否则会跑到随机模拟**
        } 
        catch (err) {
          console.error("CASSOCK 调用失败：", err)
          methodStatuses[methodName] = "red"
          stageScores['后门检测与修复'] = Math.max(0, stageScores['后门检测与修复'] - 10)
          return
        }
      }

      /* ==========================================
         🔥 非 CASSOCK → 仍然保持 Sleep 模拟
      ========================================== */
      await sleep(2000)

      const rand = Math.random()
      let result = "green"
      if (rand > 0.7) result = "yellow"
      if (rand > 0.9) result = "red"

      methodStatuses[methodName] = result
      stageScores[stageName] = Math.max(0, stageScores[stageName] - (Math.random() * 3 + 1))
    }



    /* ==========================================
       🔥 新逻辑：所有结构调用同一个 executeMethod
    ========================================== */
    if (isNested) {
      for (const subStageName of Object.keys(methods)) {
        // 特殊处理模型训练阶段，让其方法同时执行
        if (subStageName === "模型训练阶段") {
          const trainMethods = Object.keys(methods[subStageName]);
          // 同时将所有方法设置为running状态
          trainMethods.forEach(methodName => {
            currentMethod.value = methodName;
            methodStatuses[methodName] = "running";
          });
          
          // 模拟执行过程
          await sleep(2000);
          
          // 统一生成一个随机结果
          const rand = Math.random();
          let result = "green";
          if (rand > 0.7) result = "yellow";
          if (rand > 0.9) result = "red";
          
          // 同时将所有方法设置为相同的完成状态
          trainMethods.forEach(methodName => {
            methodStatuses[methodName] = result;
          });
          
          // 计算减分（只减一次）
          stageScores[stageName] = Math.max(0, stageScores[stageName] - (Math.random() * 3 + 1));
        } else {
          // 其他子阶段保持原有逻辑
          for (const methodName of Object.keys(methods[subStageName])) {
            await executeMethod(methodName, stageName);
          }
        }
      }
    } else {
      for (const methodName of Object.keys(methods)) {
        await executeMethod(methodName, stageName);
      }
    }
  }

  isEvaluating.value = false
  currentMethod.value = ""
  currentStage.value = "完成"
}


onMounted(() => {
  runSimulation()
})
</script>

<style>
@keyframes wave {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

.animate-wave {
  animation: wave 3s linear infinite;
}

/* Custom scrollbar */
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: #e5e7eb;
  border-radius: 20px;
}
</style>
