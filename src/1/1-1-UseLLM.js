/**
 * Thought-Action Agent
 */

import axios from 'axios';
import OpenAI from 'openai';
import { TavilyClient } from 'tavily';
import { tavilyConfigKey } from '../../config/tavily.config.js';
import { llmConfig } from '../../config/llm.config.js';

// 初始化 OpenAI 客户端
const openai = new OpenAI({
    apiKey: llmConfig.apiKey,
    baseURL: llmConfig.baseURL,
});

// step 1：prompt
const AGENT_SYSTEM_PROMPT = `你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- \`get_weather(city: str)\`: 查询指定城市的实时天气。
- \`get_attraction(city: str, weather: str)\`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在\`Action:\`字段后使用 \`finish(answer="...")\` 来输出最终答案。

请开始吧！`;

// step 2：tools list
async function get_weather({ city }) {
    const url = `https://wttr.in/${encodeURIComponent(city)}?format=j1`;

    try {
        const response = await axios.get(url);
        const data = response.data;

        const current = data.current_condition[0];
        const weatherDesc = current.weatherDesc[0].value;
        const tempC = current.temp_C;

        return `${city}当前天气：${weatherDesc}，气温${tempC}摄氏度`;
    } catch (error) {
        return `错误：查询天气时遇到问题 - ${error.message}`;
    }
}

// 工具 2：Tavily 搜索景点

async function get_attraction({ city, weather }) {
    const apiKey = tavilyConfigKey;
    if (!apiKey) {
        return '错误：未配置TAVILY_API_KEY环境变量';
    }

    const client = new TavilyClient({ apiKey });
    const query = `'${city}' 在'${weather}'天气下最值得去的旅游景点推荐及理由`;

    try {
        const response = await client.search({
            query,
            search_depth: 'basic',
            include_answer: true,
        });

        if (response.answer) {
            return response.answer;
        }

        const results = response.results || [];
        if (results.length === 0) {
            return '抱歉，没有找到相关的旅游景点推荐。';
        }

        const formatted = results.map(r => `- ${r.title}: ${r.content}`);
        return '根据搜索，为您找到以下信息:\n' + formatted.join('\n');
    } catch (error) {
        return `错误：执行Tavily搜索时出现问题 - ${error.message}`;
    }
}

const TOOLS = {
    get_weather,
    get_attraction,
};

// step 3：agent

async function callLLM(messages) {
    const completion = await openai.chat.completions.create({
        model: llmConfig.defaultModel,
        messages: messages,
        stream: false,
    });

    return completion.choices[0].message.content;
}

// step 4：run

async function runAgent(userInput) {
    // 1. 初始化对话历史
    const promptHistory = [`用户请求: ${userInput}`];

    // 2. 循环（最多5次）
    for (let i = 0; i < 5; i++) {
        // ① 构建完整 Prompt
        const fullPrompt = promptHistory.join('\n');

        // ② 调用 LLM
        const llmOutput = await callLLM([
            { role: 'system', content: AGENT_SYSTEM_PROMPT },
            { role: 'user', content: fullPrompt },
        ]);

        // 打印 LLM 输出结构
        console.log('\n========== LLM 输出（第', i + 1, '轮）==========');
        console.log('完整输出:\n', llmOutput);
        console.log('输出类型:', typeof llmOutput);
        console.log('输出长度:', llmOutput.length);
        console.log('==========================================\n');

        // ③ 保存 LLM 输出到历史
        promptHistory.push(llmOutput);

        // ④ 解析 Action
        const actionMatch = llmOutput.match(/Action: (.*)/);
        if (!actionMatch) {
            console.log('错误：未找到 Action');
            console.log('完整输出内容:', llmOutput);
            break;
        }

        const actionStr = actionMatch[1].trim();
        console.log('解析到的 Action:', actionStr);

        // ⑤ 判断是否完成
        if (actionStr.startsWith('finish')) {
            const answerMatch = actionStr.match(/finish\(answer="(.*)"\)/);
            const finalAnswer = answerMatch ? answerMatch[1] : '任务完成';
            console.log(`最终答案: ${finalAnswer}`);
            break;
        }

        // ⑥ 解析工具名和参数
        const toolMatch = actionStr.match(/(\w+)\((.*)\)/);
        if (!toolMatch) {
            console.log('错误：无法解析工具调用');
            console.log(`Action 内容: ${actionStr}`);
            break;
        }

        const toolName = toolMatch[1];
        const argsStr = toolMatch[2];

        console.log('解析到的工具名:', toolName);
        console.log('解析到的参数字符串:', argsStr);

        // 解析参数：city="北京", weather="晴天"
        const kwargs = {};
        const argMatches = argsStr.matchAll(/(\w+)="([^"]*)"/g);
        for (const match of argMatches) {
            kwargs[match[1]] = match[2];
        }

        console.log('解析后的参数对象:', kwargs);

        // ⑦ 执行工具
        let observation;
        if (TOOLS[toolName]) {
            observation = await TOOLS[toolName](kwargs);
        } else {
            observation = `错误：未定义的工具 '${toolName}'`;
        }

        // ⑧ 保存观察结果到历史
        const observationStr = `Observation: ${observation}`;
        promptHistory.push(observationStr);

        console.log(`工具调用: ${toolName}`);
        console.log(observationStr);
    }
}

// 运行
await runAgent('帮我查询北京天气，然后推荐景点');
