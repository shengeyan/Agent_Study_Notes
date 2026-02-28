import OpenAI from 'openai';
import dotenv from 'dotenv';

// 加载环境变量
dotenv.config();

// ============ 父类：支持多供应商 ============
class LLMClient {
    constructor(model, apiKey, baseURL, provider = 'auto') {
        this.model = model || process.env.LLM_MODEL;
        this.apiKey = apiKey || process.env.LLM_API_KEY;
        this.baseURL = baseURL || process.env.LLM_BASE_URL;
        this.provider = provider || process.env.LLM_PROVIDER || 'auto';

        this.client = new OpenAI({
            apiKey: this.apiKey,
            baseURL: this.baseURL,
        });
    }

    async chat(messages) {
        const completion = await this.client.chat.completions.create({
            model: this.model,
            messages: messages,
        });
        return completion.choices[0].message.content;
    }

    /**
     * 流式输出
     * @param {Array} messages - 消息数组
     * @returns {AsyncIterator<string>}
     */
    async *chatStream(messages) {
        const stream = await this.client.chat.completions.create({
            model: this.model,
            messages: messages,
            stream: true,
        });

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
                yield content;
            }
        }
    }
}

// ============ 子类：扩展支持 ModelScope 和 Ollama ============
class MyLLM extends LLMClient {
    constructor(model, apiKey, baseURL, provider = 'auto') {
        // 🔍 步骤 1: 如果 provider='auto'，自动检测
        if (provider === 'auto') {
            provider = MyLLM._autoDetectProvider(apiKey, baseURL);
            console.log(`🔍 自动检测到 provider: ${provider}`);
        }

        // 🔧 步骤 2: 根据检测结果解析配置
        const config = MyLLM._resolveCredentials(provider, model, apiKey, baseURL);

        super(null, null, null, config.provider);

        // 应用解析后的配置
        this.provider = config.provider;
        this.apiKey = config.apiKey;
        this.baseURL = config.baseURL;
        this.model = config.model;

        console.log(`✅ LLM 初始化完成: provider=${this.provider}, model=${this.model}`);

        this.client = new OpenAI({
            apiKey: this.apiKey,
            baseURL: this.baseURL,
        });
    }

    /**
     * 🔍 自动检测 LLM 服务商
     * 优先级：特定环境变量 > base_url 域名/端口匹配 > API Key 格式
     */
    static _autoDetectProvider(apiKey, baseURL) {
        // 1️⃣ 最高优先级：检查特定服务商的环境变量
        if (process.env.MODELSCOPE_API_KEY) return 'modelscope';
        if (process.env.OPENAI_API_KEY) return 'openai';
        if (process.env.OLLAMA_BASE_URL) return 'ollama';

        // 2️⃣ 次高优先级：根据 base_url 判断
        const actualBaseURL = baseURL || process.env.LLM_BASE_URL;
        if (actualBaseURL) {
            const url = actualBaseURL.toLowerCase();

            // 云服务商域名匹配
            if (url.includes('api-inference.modelscope.cn')) return 'modelscope';
            if (url.includes('api.openai.com')) return 'openai';
            if (url.includes('open.bigmodel.cn')) return 'zhipu';

            // 本地服务端口匹配
            if (url.includes('localhost') || url.includes('127.0.0.1')) {
                if (url.includes(':11434')) return 'ollama';
                if (url.includes(':8000')) return 'vllm';
                return 'local';
            }
        }

        // 3️⃣ 辅助判断：分析 API Key 格式
        const actualApiKey = apiKey || process.env.LLM_API_KEY;
        if (actualApiKey) {
            if (actualApiKey.startsWith('ms-')) return 'modelscope';
            if (actualApiKey.startsWith('sk-')) return 'openai';
        }

        // 4️⃣ 默认返回 'auto'，使用通用配置
        return 'auto';
    }

    /**
     * 🔧 根据 provider 解析具体的配置
     */
    static _resolveCredentials(provider, model, apiKey, baseURL) {
        switch (provider) {
            case 'modelscope':
                return {
                    provider: 'modelscope',
                    apiKey: apiKey || process.env.MODELSCOPE_API_KEY || process.env.LLM_API_KEY,
                    baseURL: baseURL || process.env.LLM_BASE_URL || 'https://api-inference.modelscope.cn/v1/',
                    model: model || process.env.LLM_MODEL || 'Qwen/Qwen2.5-VL-72B-Instruct',
                };

            case 'ollama':
                return {
                    provider: 'ollama',
                    apiKey: apiKey || process.env.OLLAMA_API_KEY || 'ollama',
                    baseURL: baseURL || process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1',
                    model: model || process.env.OLLAMA_MODEL || 'llama3.2',
                };

            case 'openai':
                return {
                    provider: 'openai',
                    apiKey: apiKey || process.env.OPENAI_API_KEY || process.env.LLM_API_KEY,
                    baseURL: baseURL || process.env.LLM_BASE_URL || 'https://api.openai.com/v1',
                    model: model || process.env.LLM_MODEL || 'gpt-4',
                };

            default:
                return {
                    provider: provider,
                    apiKey: apiKey || process.env.LLM_API_KEY,
                    baseURL: baseURL || process.env.LLM_BASE_URL,
                    model: model || process.env.LLM_MODEL,
                };
        }
    }

    async chat(messages) {
        console.log(`[${this.model}] 正在调用 LLM...`);
        console.log(`使用模型: ${this.model}, baseURL: ${this.baseURL}`);
        return super.chat(messages);
    }
}

export default MyLLM;
