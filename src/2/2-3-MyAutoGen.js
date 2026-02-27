import OpenAI from 'openai';
import { llmConfig } from '../../config/llm.config.js';

// ============ Agent Client ============
function CreateAgentClient(name, basePrompt) {
    const client = new OpenAI({
        apiKey: llmConfig.apiKey,
        baseURL: llmConfig.baseURL,
    });

    return {
        name: name,
        systemPrompt: basePrompt,
        async chat(messages) {
            const stream = await client.chat.completions.create({
                model: llmConfig.defaultModel,
                messages: [{ role: 'system', content: basePrompt }, ...messages],
                stream: true, // 启用流式输出
            });

            let fullResponse = '';
            for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content || '';
                if (content) {
                    process.stdout.write(content); // 逐字打印
                    fullResponse += content;
                }
            }
            console.log(); // 换行
            return fullResponse;
        },
    };
}

// ============ Round-Robin Group Chat ============
class RoundRobinGroupChat {
    constructor(participants, terminationKeyword = 'TERMINATE', maxTurns = 20) {
        this.participants = participants;
        this.terminationKeyword = terminationKeyword;
        this.maxTurns = maxTurns;
        this.messageHistory = [];
    }

    async run(initialMessage) {
        console.log(`\n🚀 群聊开始，初始消息: "${initialMessage}"\n`);
        this.messageHistory.push({ role: 'user', content: initialMessage });
        let currentTurn = 0;

        while (currentTurn < this.maxTurns) {
            for (const agent of this.participants) {
                console.log(`\n${'='.repeat(60)}`);
                console.log(`👤 ${agent.name} 发言 (第 ${currentTurn + 1} 轮)`);
                console.log('='.repeat(60));

                const response = await agent.chat(this.messageHistory);

                console.log(); // 仅添加分隔空行

                this.messageHistory.push({
                    role: 'assistant',
                    content: `[${agent.name}]: ${response}`,
                    name: agent.name,
                });

                if (response.includes(this.terminationKeyword)) {
                    console.log(`\n✅ 检测到终止关键词 "${this.terminationKeyword}"，对话结束`);
                    return this.messageHistory;
                }

                currentTurn++;
                if (currentTurn >= this.maxTurns) {
                    console.log('\n⚠️ 达到最大轮次，对话结束');
                    return this.messageHistory;
                }
            }
        }

        return this.messageHistory;
    }
}

// ============ Main Function ============
async function run() {
    // 产品经理
    const productManager = CreateAgentClient(
        '产品经理',
        `你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后说"请工程师开始实现"。
`
    );

    // 工程师
    const engineerManager = CreateAgentClient(
        '工程师',
        `你是一位资深的软件工程师，擅长 JavaScript 开发和 Web 应用构建。

你的技术专长包括：
1. **JavaScript 编程**：熟练掌握 JavaScript/Node.js 语法和最佳实践
2. **Web 开发**：精通 React、Vue、Express 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"请代码审查员检查"。`
    );

    // 代码审查员
    const assistantManager = CreateAgentClient(
        '代码审查员',
        `你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。`
    );

    // 用户代理
    const userProxy = CreateAgentClient(
        '用户代理',
        `用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。`
    );

    // 创建群聊
    const teamChat = new RoundRobinGroupChat(
        [productManager, engineerManager, assistantManager, userProxy],
        'TERMINATE',
        20
    );

    // 启动对话
    await teamChat.run('请帮我开发一个简单的待办事项应用，需要支持添加、删除和标记完成功能。');
}

run();
