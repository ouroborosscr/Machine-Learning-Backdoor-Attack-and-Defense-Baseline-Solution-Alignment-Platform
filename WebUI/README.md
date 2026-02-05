# 环境准备



## 确保你的环境满足以下要求：

```

git: 你需要使用 git 来克隆和管理项目版本。

python: >= 3.10

NodeJS: >=18.0.0，推荐 18.19.0 或更高。

pnpm: >= 8.0.0，推荐最新版本。

```

## 安装依赖

```

pip install -r requirements.txt

cd web && pnpm i

```

## 运行前端

```

cd web && pnpm dev

```

## 运行后端

```

python run.py

```

## 运行应用模拟

```
cd test

#启动修复后模型

CUDA_VISIBLE_DEVICES=0 vllm serve /Qwen2.5-7B-Instruct \

&nbsp; --served-model-name qwen2.5b-inst-1 \

&nbsp; --port 8000 \

&nbsp; --api-key EMPTY \

&nbsp; --dtype bfloat16 \

&nbsp; --max-model-len 32768 \

&nbsp; --enable-auto-tool-choice \

&nbsp; --tool-call-parser hermes \

&nbsp; --gpu-memory-utilization 0.5



\#启动修复前模型

CUDA\_VISIBLE\_DEVICES=0 vllm serve /poison_qwen2.5_7B \

&nbsp; --served-model-name qwen2.5b-inst-1 \

&nbsp; --port 8001 \

&nbsp; --api-key EMPTY \

&nbsp; --dtype bfloat16 \

&nbsp; --max-model-len 32768 \

&nbsp; --enable-auto-tool-choice \

&nbsp; --tool-call-parser hermes \

&nbsp; --gpu-memory-utilization 0.5



#启动修复后智能体

python qwen_use_8000.py



#启动修复前智能体

python qwen_use_8001.py



#启动应用前端

python webui1.py

```



