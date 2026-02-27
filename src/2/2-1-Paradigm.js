import { llmConfig } from '../../config/llm.config.js';
import OpenAI from 'openai';
import { TavilyClient } from 'tavily';
import { tavilyConfigKey } from '../../config/tavily.config.js';
import { CLIPSegModel } from '@xenova/transformers';

// llm客服端类
class LLMClient {
    constructor() {
        this.openai = new OpenAI({
            apiKey: llmConfig.apiKey,
            baseURL: llmConfig.baseURL,
        });
    }

    async chat(messages) {
        const completion = await this.openai.chat.completions.create({
            model: llmConfig.defaultModel,
            messages: messages,
        });
        return completion.choices[0].message.content;
    }
}

// 搜索工具类
class TavilySearch {
    constructor() {
        const apiKey = tavilyConfigKey;
        if (!apiKey) {
            throw new Error('错误: TAVILY_API_KEY 未在环境变量中配置。');
        }
        this.client = new TavilyClient({ apiKey });
    }

    async search(query) {
        console.log(`🔍 正在执行 [Tavily] 网页搜索: ${query}`);

        try {
            const response = await this.client.search({
                query: query,
                search_depth: 'basic',
                include_answer: true,
                max_results: 3,
            });

            // 优先级 1: 直接答案（Tavily 的 AI 生成摘要）
            if (response.answer) {
                return response.answer;
            }

            // 优先级 2: 搜索结果摘要
            if (response.results && response.results.length > 0) {
                const snippets = response.results.map((res, i) => {
                    const title = res.title || '';
                    const content = res.content || '';
                    return `[${i + 1}] ${title}\n${content}`;
                });
                return snippets.join('\n\n');
            }

            return `对不起，没有找到关于 '${query}' 的信息。`;
        } catch (error) {
            return `搜索时发生错误: ${error.message}`;
        }
    }
}

// 工具类
class ToolExecutor {
    /**
     * 一个工具执行器，负责管理和执行工具。
     */
    constructor() {
        this.tools = new Map();
    }

    // 注册工具
    registerTool(name, description, func) {
        if (this.tools.has(name)) {
            console.log(`警告: 工具 '${name}' 已存在，将被覆盖。`);
        }
        this.tools.set(name, { description, func });
        console.log(`工具 '${name}' 已注册。`);
    }

    // 获取工具
    getTool(name) {
        const tool = this.tools.get(name);
        return tool ? tool.func : null;
    }

    // 所有工具的列表
    getAvailableTools() {
        const toolList = Array.from(this.tools.entries()).map(([name, info]) => `- ${name}: ${info.description}`);
        return toolList.join('\n');
    }

    // 执行工具
    async executeTool(name, args) {
        const func = this.getTool(name);
        if (!func) {
            throw new Error(`工具 '${name}' 不存在`);
        }
        return await func(args);
    }
}

/**
 * ReAct 智能体
 * Thought-Action
 */
class ReActAgent {
    constructor(llmClient, toolExecutor, maxSteps = 5) {
        this.llmClient = llmClient;
        this.toolExecutor = toolExecutor;
        this.maxSteps = maxSteps;
        this.history = [];

        // Prompt 模板
        this.REACT_PROMPT_TEMPLATE = `请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- \`工具名称[工具参数]\`: 调用一个可用工具。
- \`Finish[最终答案]\`: 当你认为已经获得最终答案时。

现在，请开始解决以下问题:
Question: {question}
History: {history}`;
    }

    /**
     * 运行 ReAct 智能体来回答一个问题
     */
    async run(question) {
        this.history = []; // 每次运行时重置历史记录
        let currentStep = 0;

        while (currentStep < this.maxSteps) {
            currentStep++;
            console.log(`\n--- 第 ${currentStep} 步 ---`);

            // 1. 格式化提示词
            const toolsDesc = this.toolExecutor.getAvailableTools();
            const historyStr = this.history.join('\n');
            const prompt = this.REACT_PROMPT_TEMPLATE.replace('{tools}', toolsDesc)
                .replace('{question}', question)
                .replace('{history}', historyStr);

            // 2. 调用 LLM 进行思考
            const messages = [{ role: 'user', content: prompt }];

            console.log('⏳ 正在调用 LLM...');
            let responseText;
            try {
                responseText = await this.llmClient.chat(messages);
            } catch (error) {
                console.log(`❌ LLM 调用失败: ${error.message}`);
                break;
            }

            if (!responseText) {
                console.log('错误: LLM 未能返回有效响应。');
                break;
            }

            console.log(`\n🤖 LLM 响应:\n${responseText}\n`);

            // 3. 解析 LLM 的输出
            const { thought, action } = this._parseOutput(responseText);

            if (thought) {
                console.log(`💭 思考: ${thought}`);
            }

            if (!action) {
                console.log('警告: 未能解析出有效的 Action，流程终止。');
                break;
            }

            console.log(`\n⚡ 解析到的 Action: "${action}"`); // 添加调试输出

            // 4. 执行 Action
            if (action.startsWith('Finish')) {
                // 如果是 Finish 指令，提取最终答案并结束
                console.log('🔍 检测到 Finish 指令');

                // 使用 [\s\S] 匹配包括换行符在内的所有字符
                const finishMatch = action.match(/Finish\[([\s\S]*)\]/);

                if (finishMatch) {
                    const finalAnswer = finishMatch[1].trim();
                    console.log(`\n🎉 最终答案:\n${finalAnswer}`);
                    return finalAnswer;
                } else {
                    console.log(`⚠️ 警告: 无法匹配 Finish[...] 格式`);
                    console.log(`   测试是否包含 '[': ${action.includes('[')}`);
                    console.log(`   测试是否包含 ']': ${action.includes(']')}`);

                    break;
                }
            }

            const { toolName, toolInput } = this._parseAction(action);
            if (!toolName || !toolInput) {
                console.log('警告: 无效的 Action 格式，流程终止。');
                break;
            }

            console.log(`🎬 行动: ${toolName}[${toolInput}]`);

            // 调用工具
            let observation;
            const toolFunction = this.toolExecutor.getTool(toolName);
            if (!toolFunction) {
                observation = `错误: 未找到名为 '${toolName}' 的工具。`;
            } else {
                try {
                    observation = await toolFunction(toolInput); // 调用真实工具
                } catch (error) {
                    observation = `错误: 工具执行失败 - ${error.message}`;
                }
            }

            console.log(`👀 观察: ${observation}`);

            // 5. 将本轮的 Action 和 Observation 添加到历史记录中
            this.history.push(`Action: ${action}`);
            this.history.push(`Observation: ${observation}`);
        }

        // 循环结束
        console.log('\n已达到最大步数，流程终止。');
        return null;
    }

    /**
     * 解析 LLM 的输出，提取 Thought 和 Action
     */
    _parseOutput(text) {
        // 匹配 Thought: 后面的内容，直到遇到 Action: 或字符串结尾
        const thoughtMatch = text.match(/Thought:\s*(.+?)(?=\nAction:|$)/s);

        // 匹配 Action: 后面的内容，直到遇到 Thought: 或字符串结尾
        const actionMatch = text.match(/Action:\s*([\s\S]+?)(?=\nThought:|$)/);

        const thought = thoughtMatch ? thoughtMatch[1].trim() : null;
        const action = actionMatch ? actionMatch[1].trim() : null;

        return { thought, action };
    }

    /**
     * 解析 Action 字符串，提取工具名称和输入
     */
    _parseAction(actionText) {
        const match = actionText.match(/(\w+)\[(.+?)\]/);
        if (match) {
            return {
                toolName: match[1],
                toolInput: match[2],
            };
        }
        return { toolName: null, toolInput: null };
    }
}

/**
 * Plan-and-Solve 智能体
 */
class PlanAndSolveAgent {
    constructor(llmClient, toolExecutor) {
        this.llmClient = llmClient;
        this.toolExecutor = toolExecutor;
        this.PROMPT = `
        你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个JavaScript数组，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，\`\`\`javascript 与 \`\`\` 作为前后缀是必要的:
\`\`\`javascript
["步骤1", "步骤2", "步骤3", ...]
\`\`\`
`;
        this.PLAN_PROMPT = ` EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

`;
    }

    async plan(question) {
        const prompt = this.PROMPT.replace('{question}', question);

        // 构建消息列表
        const messages = [{ role: 'user', content: prompt }];

        console.log('\n--- 正在生成计划 ---');

        // 调用 LLM 生成计划
        const responseText = await this.llmClient.chat(messages);

        if (!responseText) {
            console.log('❌ LLM 未返回有效响应');
            return [];
        }

        console.log(`✅ 计划已生成:\n${responseText}\n`);

        // 解析 LLM 输出的列表字符串
        try {
            // 找到 ```javascript 和 ``` 之间的内容
            const parts = responseText.split('```javascript');
            if (parts.length < 2) {
                throw new Error('未找到 ```javascript 代码块');
            }

            const planStr = parts[1].split('```')[0].trim();

            // 使用 JSON.parse 将字符串转换为数组
            const plan = JSON.parse(planStr);

            if (!Array.isArray(plan)) {
                throw new Error('解析结果不是数组');
            }
            console.log(`✅ 计划: ${plan}`);

            // 自动运行计划
            return await this.execute(question, plan);

            // return plan;
        } catch (error) {
            console.log(`❌ 解析计划时出错: ${error.message}`);
            console.log(`原始响应: ${responseText}`);
            return [];
        }
    }

    async execute(question, plans) {
        let history = '';
        let response;

        for (const plan of plans) {
            console.log(`\n--- 正在执行计划: ${plan} ---`);

            const prompt = this.PLAN_PROMPT.replace('{question}', question)
                .replace('{plan}', plans)
                .replace('{history}', history)
                .replace('{current_step}', plan);

            response = await this.llmClient.chat([
                {
                    role: 'user',
                    content: prompt,
                },
            ]);

            console.log(`✅ 当前结果: ${response}`);
        }
        console.log(`\n--------------`);

        console.log(`✅ 最终结果: ${response}`);
        return response;
    }
}

// 记忆类
class Memory {
    constructor() {
        this.records = [];
    }

    addRecord(recordType, content) {
        const record = {
            type: recordType,
            content: content,
        };
        this.records.push(record);
        console.log(`📝 记忆已更新，新增一条 '${recordType}' 记录。`);
    }

    getTrajectory() {
        const trajectoryParts = [];

        for (const record of this.records) {
            if (record.type === 'execution') {
                trajectoryParts.push(`--- 上一轮尝试 (代码) ---\n${record.content}`);
            } else if (record.type === 'reflection') {
                trajectoryParts.push(`--- 评审员反馈 ---\n${record.content}`);
            }
        }

        return trajectoryParts.join('\n\n');
    }

    getLastExecution() {
        // 从后往前遍历
        for (let i = this.records.length - 1; i >= 0; i--) {
            if (this.records[i].type === 'execution') {
                return this.records[i].content;
            }
        }
        return null;
    }
}

// Reflection 智能体
class Reflection {
    constructor(question, client, maxSteps = 3) {
        this.question = question;
        this.client = client;
        this.maxSteps = maxSteps;
        this.ExecutionPrompt = `你是一位资深的javascript程序员。请根据以下要求，编写一个javascript函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。`;
        this.ReflectionPrompt = `
        你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下javascript代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
\`\`\`javascript
{code}
\`\`\`

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
`;
        this.RefinementPrompt = `
你是一位资深的javascript程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:

{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
`;
    }

    async run() {
        const memory = new Memory();
        let prompt = this.ExecutionPrompt.replace('{task}', this.question);
        let reflectionResponse = ''; // 用于存储评审响应

        while (!reflectionResponse.includes('无需改进')) {
            // 1. 生成代码 - execution
            let executionResponse = await this._get_llm_response(prompt);
            memory.addRecord('execution', executionResponse);
            console.log('\n🔧 执行响应（代码）：\n', executionResponse);
            console.log('-------------------------------------');

            // 2. 评审代码 - reflection
            prompt = this.ReflectionPrompt.replace('{task}', this.question).replace(
                '{code}',
                memory.getLastExecution()
            );
            reflectionResponse = await this._get_llm_response(prompt);
            memory.addRecord('reflection', reflectionResponse);
            console.log('\n🔍 评审响应：\n', reflectionResponse);
            console.log('-------------------------------------');

            // 3. 检查是否需要改进
            if (reflectionResponse.includes('无需改进')) {
                console.log('\n✅ 代码已通过评审，无需进一步改进！');
                break;
            }

            // 4. 优化代码 - refinement
            console.log('\n🔄 需要改进，正在优化代码...');
            prompt = this.RefinementPrompt.replace('{task}', this.question)
                .replace('{last_code_attempt}', memory.getLastExecution())
                .replace('{feedback}', reflectionResponse);
            let refinementResponse = await this._get_llm_response(prompt);
            memory.addRecord('execution', refinementResponse); // 优化后的代码也是 execution
            console.log('\n✨ 优化响应（新代码）：\n', refinementResponse);
            console.log('-------------------------------------');

            // 下一轮循环：用新代码重新评审
        }

        console.log('\n========== 最终代码 ==========');
        const finalCode = memory.getLastExecution();
        console.log(finalCode);
        return finalCode;
    }

    async _get_llm_response(prompt) {
        const response = await this.client.chat([
            {
                role: 'user',
                content: prompt,
            },
        ]);

        return response || '请求错误';
    }
}

// 测试
async function run() {
    const client = new LLMClient();
    const toolExecutor = new ToolExecutor();

    // 注册搜索工具
    toolExecutor.registerTool('search', '搜索网页内容', async args => {
        const search = new TavilySearch();
        return await search.search(args);
    });

    // ------------------------------------------------
    // 运行 agent
    // const agent = new ReActAgent(client, toolExecutor, 5);
    // const answer = await agent.run('英伟达最新的GPU型号是什么？');

    // console.log('\n========== 最终结果 ==========');
    // console.log(answer || '未能获得答案');

    // ------------------------------------------------
    // 运行 plan-and-solve agent
    const planAndsolveAgent = new PlanAndSolveAgent(client, toolExecutor);
    await planAndsolveAgent.plan(
        '一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？'
    );

    // ------------------------------------------------
    // 运行 Reflection
    // const reflection = new Reflection('编写一个javascript函数，找出1到n之间所有的素数 (prime numbers)', client);
    // reflection.run();
}

run();
