import 'dotenv/config';
import MyLLM from '../2/2-4-HelloAgent/MyLLM.js';
import SimpleAgent from '../2/2-4-HelloAgent/SimpleAgent.js';

// 模拟 ToolRegistry 和 CalculatorTool（需要根据实际项目调整）
class CalculatorTool {
    constructor() {
        this.name = 'calculator';
        this.description = '数学计算工具，可以执行基本的数学运算';
    }

    run(params) {
        try {
            // 如果 params 是字符串，直接计算
            const expression = typeof params === 'string' ? params : params.input || params.expression;
            // 使用 eval 计算表达式（注意：生产环境中应使用更安全的方式）
            const result = eval(expression);
            return `计算结果: ${result}`;
        } catch (e) {
            return `计算错误: ${e.message}`;
        }
    }
}

class ToolRegistry {
    constructor() {
        this.tools = new Map();
    }

    registerTool(tool) {
        this.tools.set(tool.name, tool);
    }

    getTool(name) {
        return this.tools.get(name);
    }

    executeTool(name, params) {
        const tool = this.tools.get(name);
        if (!tool) {
            throw new Error(`工具 ${name} 不存在`);
        }
        return tool.run(params);
    }

    getToolsDescription() {
        if (this.tools.size === 0) {
            return '暂无可用工具';
        }
        const descriptions = [];
        for (const [name, tool] of this.tools.entries()) {
            descriptions.push(`- ${name}: ${tool.description}`);
        }
        return descriptions.join('\n');
    }

    unregister(name) {
        return this.tools.delete(name);
    }

    listTools() {
        return Array.from(this.tools.keys());
    }
}

// 主测试函数
async function main() {
    // 创建 LLM 实例（注意参数顺序：model, apiKey, baseURL, provider）
    const llm = new MyLLM(
        'moonshotai/kimi-k2.5', // model
        process.env.LLM_API_KEY, // apiKey
        process.env.LLM_BASE_URL, // baseURL
        'openai' // provider（兼容 OpenAI API 的服务）
    );

    // 测试1: 基础对话 Agent（无工具）
    console.log('=== 测试1:基础对话 ===');
    const basicAgent = new SimpleAgent('基础助手', llm, '你是一个友好的AI助手，请用简洁明了的方式回答问题。');

    const response1 = await basicAgent.run('你好，请介绍一下自己');
    console.log(`基础对话响应: ${response1}\n`);

    // 测试2: 带工具的 Agent
    console.log('=== 测试2:工具增强对话 ===');
    const toolRegistry = new ToolRegistry();
    const calculator = new CalculatorTool();
    toolRegistry.registerTool(calculator);

    const enhancedAgent = new SimpleAgent(
        '增强助手',
        llm,
        '你是一个智能助手，可以使用工具来帮助用户。',
        null,
        toolRegistry,
        true
    );

    const response2 = await enhancedAgent.run('请帮我计算 15 * 8 + 32');
    console.log(`工具增强响应: ${response2}\n`);

    // 测试3: 流式响应
    console.log('=== 测试3:流式响应 ===');
    process.stdout.write('流式响应: ');
    for await (const chunk of basicAgent.streamRun('请解释什么是人工智能')) {
        // 内容已在 streamRun 中实时打印
    }

    // 测试4: 动态添加工具
    console.log('\n=== 测试4:动态工具管理 ===');
    console.log(`添加工具前: ${basicAgent.hasTools()}`);

    // 为 basicAgent 添加 toolRegistry
    basicAgent.toolRegistry = new ToolRegistry();
    basicAgent.addTool(calculator);

    console.log(`添加工具后: ${basicAgent.hasTools()}`);
    console.log(`可用工具: ${basicAgent.listTools()}`);

    // 查看对话历史
    console.log(`\n对话历史: ${basicAgent.getHistory().length} 条消息`);
}

// 运行测试
main().catch(console.error);
