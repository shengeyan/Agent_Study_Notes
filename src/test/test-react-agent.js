import 'dotenv/config';
import MyLLM from '../2/2-4-HelloAgent/MyLLM.js';
import ReActAgent from '../2/2-4-HelloAgent/ReActAgent.js';

// 工具类定义
class CalculatorTool {
    constructor() {
        this.name = 'calculator';
        this.description = '数学计算工具，可以执行基本的数学运算（如：2+3, 15*8+32）';
    }

    run(params) {
        try {
            const expression = typeof params === 'string' ? params : params.input || params.expression;
            const result = eval(expression);
            return `计算结果: ${result}`;
        } catch (e) {
            return `计算错误: ${e.message}`;
        }
    }
}

class SearchTool {
    constructor() {
        this.name = 'search';
        this.description = '搜索工具，可以搜索互联网上的信息（输入搜索关键词）';
    }

    run(params) {
        const query = typeof params === 'string' ? params : params.query;
        // 模拟搜索结果
        const mockResults = {
            'Python': 'Python是一种高级编程语言，由Guido van Rossum在1991年创建。它以简洁的语法和强大的功能而闻名。',
            'JavaScript': 'JavaScript是一种高级的、解释型编程语言，是Web开发的核心技术之一。',
            '人工智能': '人工智能（AI）是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统。',
            '默认': `搜索结果：关于"${query}"的信息 - 这是一个模拟的搜索结果。`
        };
        
        for (const [keyword, result] of Object.entries(mockResults)) {
            if (query.includes(keyword)) {
                return result;
            }
        }
        return mockResults['默认'];
    }
}

class WeatherTool {
    constructor() {
        this.name = 'weather';
        this.description = '天气查询工具，可以查询指定城市的天气（输入城市名称）';
    }

    run(params) {
        const city = typeof params === 'string' ? params : params.city;
        // 模拟天气数据
        const mockWeather = {
            '北京': '北京今天晴，气温15-25℃，空气质量良好',
            '上海': '上海今天多云，气温18-26℃，有轻度雾霾',
            '深圳': '深圳今天阴转小雨，气温22-28℃，湿度较大',
            '默认': `${city}今天天气：晴，气温20-28℃`
        };
        return mockWeather[city] || mockWeather['默认'];
    }
}

// 工具注册表
class ToolRegistry {
    constructor() {
        this.tools = new Map();
    }

    registerTool(tool) {
        this.tools.set(tool.name, tool);
        console.log(`✅ 注册工具: ${tool.name}`);
    }

    getTool(name) {
        return this.tools.get(name);
    }

    executeTool(name, params) {
        const tool = this.tools.get(name);
        if (!tool) {
            return `❌ 错误：工具 "${name}" 不存在。可用工具: ${Array.from(this.tools.keys()).join(', ')}`;
        }
        console.log(`🔧 执行工具: ${name}，参数: ${params}`);
        const result = tool.run(params);
        console.log(`📦 工具返回: ${result}`);
        return result;
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

    listTools() {
        return Array.from(this.tools.keys());
    }
}

// 测试函数
async function testBasicReAct() {
    console.log('\n════════════════════════════════════════');
    console.log('📋 测试1: 基础 ReAct - 数学计算');
    console.log('════════════════════════════════════════');

    const llm = new MyLLM(
        'moonshotai/kimi-k2.5',
        process.env.LLM_API_KEY,
        process.env.LLM_BASE_URL,
        'openai'
    );

    const toolRegistry = new ToolRegistry();
    toolRegistry.registerTool(new CalculatorTool());

    const agent = new ReActAgent(
        'ReAct计算助手',
        llm,
        toolRegistry,
        null,
        null,
        5
    );

    const result = await agent.run('请计算 (25 + 13) * 4 - 8 的结果');
    console.log(`\n📊 最终答案: ${result}`);
}

async function testMultiTools() {
    console.log('\n════════════════════════════════════════');
    console.log('📋 测试2: 多工具 ReAct - 搜索+计算');
    console.log('════════════════════════════════════════');

    const llm = new MyLLM(
        'moonshotai/kimi-k2.5',
        process.env.LLM_API_KEY,
        process.env.LLM_BASE_URL,
        'openai'
    );

    const toolRegistry = new ToolRegistry();
    toolRegistry.registerTool(new CalculatorTool());
    toolRegistry.registerTool(new SearchTool());
    toolRegistry.registerTool(new WeatherTool());

    const agent = new ReActAgent(
        'ReAct多功能助手',
        llm,
        toolRegistry,
        null,
        null,
        8
    );

    const result = await agent.run('搜索一下Python编程语言，然后告诉我如果一个Python开发者每天工作8小时，一周工作5天，那么一个月工作多少小时？');
    console.log(`\n📊 最终答案: ${result}`);
}

async function testComplexReasoning() {
    console.log('\n════════════════════════════════════════');
    console.log('📋 测试3: 复杂推理 - 多步骤问题');
    console.log('════════════════════════════════════════');

    const llm = new MyLLM(
        'moonshotai/kimi-k2.5',
        process.env.LLM_API_KEY,
        process.env.LLM_BASE_URL,
        'openai'
    );

    const toolRegistry = new ToolRegistry();
    toolRegistry.registerTool(new CalculatorTool());
    toolRegistry.registerTool(new SearchTool());
    toolRegistry.registerTool(new WeatherTool());

    const agent = new ReActAgent(
        'ReAct推理专家',
        llm,
        toolRegistry,
        null,
        null,
        10
    );

    const result = await agent.run('查询北京的天气，如果温度超过20度，计算25*3+15的结果；如果不超过，搜索人工智能的信息');
    console.log(`\n📊 最终答案: ${result}`);
}

async function testMaxStepsLimit() {
    console.log('\n════════════════════════════════════════');
    console.log('📋 测试4: 最大步数限制测试');
    console.log('════════════════════════════════════════');

    const llm = new MyLLM(
        'moonshotai/kimi-k2.5',
        process.env.LLM_API_KEY,
        process.env.LLM_BASE_URL,
        'openai'
    );

    const toolRegistry = new ToolRegistry();
    toolRegistry.registerTool(new CalculatorTool());

    const agent = new ReActAgent(
        'ReAct限制测试',
        llm,
        toolRegistry,
        null,
        null,
        2  // 只允许2步
    );

    const result = await agent.run('计算 100 + 200，然后再乘以 3，最后加上 50');
    console.log(`\n📊 最终答案: ${result}`);
}

// 主函数
async function main() {
    console.log('\n🚀 开始 ReActAgent 测试套件\n');

    try {
        // 测试1: 基础计算
        await testBasicReAct();

        // 等待3秒避免 API 限流
        console.log('\n⏳ 等待 3 秒...\n');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // 测试2: 多工具协作
        await testMultiTools();

        // 等待3秒
        console.log('\n⏳ 等待 3 秒...\n');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // 测试3: 复杂推理
        await testComplexReasoning();

        // 等待3秒
        console.log('\n⏳ 等待 3 秒...\n');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // 测试4: 最大步数限制
        await testMaxStepsLimit();

        console.log('\n✅ 所有测试完成！');
    } catch (error) {
        console.error('\n❌ 测试失败:', error.message);
        console.error(error.stack);
    }
}

// 运行测试
main().catch(console.error);
