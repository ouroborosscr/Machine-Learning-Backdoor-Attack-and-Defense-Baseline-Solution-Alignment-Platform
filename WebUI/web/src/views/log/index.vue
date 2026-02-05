<script setup lang="ts">
import { ref } from 'vue';
import logData from './logs.json';

// 定义日志数据的类型接口
interface LogItem {
  id: string;
  数据集: string;
  模型结构: string;
  开始时间: string;
  完成时间: string;
  评估项目: string[];
  下载报告: string;
  应用测试: string;
  导出API: string;
  导出安全模型: string;
  导出安全数据: string;
}

// 将 JSON 对象转换为数组以便在表格中展示
const logs = ref<LogItem[]>(Object.values(logData));

// 执行操作处理函数
const handleAction = (action: string, url: string, id: string) => {
  console.log(`执行操作: ${action}, ID: ${id}, URL: ${url}`);
  
  // 当操作为"应用测试"时，打开对应的URL
  if (action === '应用测试' && url) {
    window.open(url, '_blank');
  }
  
  // 当操作为"下载报告"时，处理不同格式的URL
  else if (action === '下载报告' && url) {
    // 判断URL格式并处理
    if (url.startsWith('http://') || url.startsWith('https://')) {
      // 如果是完整的API URL，直接打开
      window.open(url, '_blank');
    } else if (url.startsWith('./')) {
      // 如果是相对路径，构造完整的API URL
      // 提取文件名（去掉开头的./）
      const fileName = url.substring(2);
      // 构造下载API URL
      const apiUrl = `/api/v1/system-manage/file-download/download?name=${encodeURIComponent(fileName)}`;
      window.open(apiUrl, '_blank');
    } else {
      // 其他情况，直接尝试打开
      window.open(url, '_blank');
    }
  }
  
  // 其他操作处理...
  // 导出API、导出模型、导出数据等操作可以根据需要类似实现
};

// 评估项目的标签颜色映射（可选，增加视觉区分）
const getTagType = (index: number) => {
  const types = ['primary', 'success', 'info', 'warning'];
  return types[index % types.length];
};
</script>

<template>
  <div class="h-full overflow-hidden flex flex-col p-4">
    <div class="bg-white dark:bg-[#18181c] rounded-xl shadow-sm flex-1 flex flex-col overflow-hidden border border-gray-200 dark:border-gray-700">
      <!-- 头部标题 -->
      <div class="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
        <h2 class="text-lg font-medium text-gray-800 dark:text-white">评估日志列表</h2>
        <div class="text-sm text-gray-500">共 {{ logs.length }} 条记录</div>
      </div>

      <!-- 表格区域 -->
      <div class="flex-1 overflow-auto">
        <table class="w-full text-left border-collapse">
          <thead class="bg-gray-50 dark:bg-[#2c2c32] sticky top-0 z-10">
            <tr>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm whitespace-nowrap">ID</th>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm whitespace-nowrap">数据集</th>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm whitespace-nowrap">模型结构</th>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm whitespace-nowrap">时间信息</th>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm min-w-[300px]">评估项目</th>
              <th class="p-4 font-medium text-gray-600 dark:text-gray-300 text-sm text-right whitespace-nowrap">操作</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 dark:divide-gray-700">
            <tr 
              v-for="item in logs" 
              :key="item.id" 
              class="hover:bg-gray-50 dark:hover:bg-[#26262a] transition-colors"
            >
              <td class="p-4 text-sm text-gray-600 dark:text-gray-300 font-mono">{{ item.id }}</td>
              <td class="p-4 text-sm text-gray-800 dark:text-gray-200 font-medium">{{ item['数据集'] }}</td>
              <td class="p-4 text-sm text-gray-600 dark:text-gray-300">{{ item['模型结构'] }}</td>
              <td class="p-4 text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">
                <div class="flex flex-col gap-1">
                  <span class="text-xs">始: {{ item['开始时间'] }}</span>
                  <span class="text-xs">终: {{ item['完成时间'] }}</span>
                </div>
              </td>
              <td class="p-4">
                <div class="flex flex-wrap gap-1.5">
                  <span 
                    v-for="(tag, idx) in item['评估项目']" 
                    :key="tag"
                    class="px-2 py-0.5 rounded text-xs border bg-gray-100 border-gray-200 text-gray-600 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-300"
                  >
                    {{ tag }}
                  </span>
                </div>
              </td>
              <td class="p-4 text-right">
                <div class="flex flex-col gap-2 items-end">
                  <div class="flex gap-2">
                    <button 
                      class="px-3 py-1 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors shadow-sm"
                      @click="handleAction('下载报告', item['下载报告'], item.id)"
                    >
                      下载报告
                    </button>
                    <button 
                      class="px-3 py-1 text-xs font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded transition-colors"
                      @click="handleAction('应用测试', item['应用测试'], item.id)"
                    >
                      应用测试
                    </button>
                  </div>
                  <div class="flex gap-2">
                    <button 
                      class="px-2 py-1 text-xs text-gray-600 hover:text-blue-600 transition-colors"
                      @click="handleAction('导出API', item['导出API'], item.id)"
                    >
                      导出API
                    </button>
                    <button 
                      class="px-2 py-1 text-xs text-gray-600 hover:text-blue-600 transition-colors"
                      @click="handleAction('导出安全模型', item['导出安全模型'], item.id)"
                    >
                      导出模型
                    </button>
                    <button 
                      class="px-2 py-1 text-xs text-gray-600 hover:text-blue-600 transition-colors"
                      @click="handleAction('导出安全数据', item['导出安全数据'], item.id)"
                    >
                      导出数据
                    </button>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 
  Fast Soybean (Soybean Admin) 风格通常使用 UnoCSS/Tailwind CSS。
  这里使用了标准的 Tailwind 类名，确保在任何支持 Tailwind 的 Vue 项目中都能正常显示。
  如果您的项目使用了 Naive UI，可以将原生 HTML 元素替换为 <n-card>, <n-data-table>, <n-button>, <n-tag> 等组件。
*/
</style>
