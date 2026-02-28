import Agent from './Agent.js';
import Message from './Message.js';

/**
 * 重写的简单对话 Agent
 * 展示如何基于框架基类构建自定义 Agent
 */
class SimpleAgent extends Agent {
    constructor(name, llm, systemPrompt = null, config = null, toolRegistry = null, enableToolCalling = true) {
        super(name, llm, systemPrompt, config);

        this.toolRegistry = toolRegistry;
        this.enableToolCalling = enableToolCalling && toolRegistry !== null;

        console.log(`✅ ${name} 初始化完成，工具调用: ${this.enableToolCalling ? '启用' : '禁用'}`);
    }

    /** 重写的运行方法 - 实现简单对话逻辑，支持可选工具调用 */
    async run(inputText, maxToolIterations = 3, kwargs = {}) {
        console.log(`🤖 ${this.name} 正在处理: ${inputText}`);

        // 构建消息列表
        const messages = [];

        // 添加系统消息（可能包含工具信息）
        const enhancedSystemPrompt = this._getEnhancedSystemPrompt();
        messages.push({ role: 'system', content: enhancedSystemPrompt });

        // 添加历史消息
        for (const msg of this._history) {
            messages.push({ role: msg.role, content: msg.content });
        }

        // 添加当前用户消息
        messages.push({ role: 'user', content: inputText });

        // 如果没有启用工具调用，使用简单对话逻辑
        if (!this.enableToolCalling) {
            const response = await this.llm.chat(messages);
            this.addMessage(new Message('user', inputText));
            this.addMessage(new Message('assistant', response));
            console.log(`✅ ${this.name} 响应完成`);
            return response;
        }

        // 支持多轮工具调用的逻辑
        return await this._runWithTools(messages, inputText, maxToolIterations, kwargs);
    }

    /** 构建增强的系统提示词，包含工具信息 */
    _getEnhancedSystemPrompt() {
        const basePrompt = this.systemPrompt || '你是一个有用的AI助手。';

        if (!this.enableToolCalling || !this.toolRegistry) {
            return basePrompt;
        }

        const toolsDescription = this.toolRegistry.getToolsDescription();
        if (!toolsDescription || toolsDescription === '暂无可用工具') {
            return basePrompt;
        }

        return `${basePrompt}

            ## 可用工具
            你可以使用以下工具来帮助回答问题:
            ${toolsDescription}

            ## 工具调用格式
            当需要使用工具时，请使用以下格式:
            \`[TOOL_CALL:{tool_name}:{parameters}]\`
            例如:\`[TOOL_CALL:search:Python编程]\` 或 \`[TOOL_CALL:memory:recall=用户信息]\`

            工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。
            `;
    }

    /** 支持工具调用的运行逻辑 */
    async _runWithTools(messages, inputText, maxToolIterations) {
        let currentIteration = 0;
        let finalResponse = '';

        while (currentIteration < maxToolIterations) {
            // 调用 LLM
            const response = await this.llm.chat(messages);

            // 检查是否有工具调用
            const toolCalls = this._parseToolCalls(response);

            if (toolCalls.length > 0) {
                console.log(`🔧 检测到 ${toolCalls.length} 个工具调用`);
                // 执行所有工具调用并收集结果
                const toolResults = [];
                let cleanResponse = response;

                // Q：为什么要移除标记
                // A：为了让content内容更加赶紧（PS:有可能回导致句意缺失，如：xxx 工具 xxx 变成 xxx xxx）
                for (const call of toolCalls) {
                    const result = this._executeToolCall(call.toolName, call.parameters);
                    toolResults.push(result);
                    // 从响应中移除工具调用标记
                    cleanResponse = cleanResponse.replace(call.original, '');
                }

                // 构建包含工具结果的消息
                messages.push({ role: 'assistant', content: cleanResponse });

                // 添加工具结果
                const toolResultsText = toolResults.join('\n\n');
                messages.push({
                    role: 'user',
                    content: `工具执行结果:\n${toolResultsText}\n\n请基于这些结果给出完整的回答。`,
                });

                currentIteration++;
                continue;
            }

            // 没有工具调用，这是最终回答
            finalResponse = response;
            break;
        }

        // 如果超过最大迭代次数，获取最后一次回答
        if (currentIteration >= maxToolIterations && !finalResponse) {
            finalResponse = await this.llm.chat(messages);
        }

        // 保存到历史记录
        this.addMessage(new Message('user', inputText));
        this.addMessage(new Message('assistant', finalResponse));
        console.log(`✅ ${this.name} 响应完成`);

        return finalResponse;
    }

    /** 解析文本中的工具调用 */
    _parseToolCalls(text) {
        const pattern = /\[TOOL_CALL:([^:]+):([^\]]+)\]/g;
        const toolCalls = [];
        let match;

        while ((match = pattern.exec(text)) !== null) {
            toolCalls.push({
                toolName: match[1].trim(),
                parameters: match[2].trim(),
                original: match[0],
            });
        }

        return toolCalls;
    }

    /** 执行工具调用 */
    _executeToolCall(toolName, parameters) {
        if (!this.toolRegistry) {
            return `❌ 错误:未配置工具注册表`;
        }

        try {
            // 智能参数解析
            if (toolName === 'calculator') {
                // 计算器工具直接传入表达式
                const result = this.toolRegistry.executeTool(toolName, parameters);
                return `🔧 工具 ${toolName} 执行结果:\n${result}`;
            } else {
                // 其他工具使用智能参数解析
                const paramDict = this._parseToolParameters(toolName, parameters);
                const tool = this.toolRegistry.getTool(toolName);
                if (!tool) {
                    return `❌ 错误:未找到工具 '${toolName}'`;
                }
                const result = tool.run(paramDict);
                return `🔧 工具 ${toolName} 执行结果:\n${result}`;
            }
        } catch (e) {
            return `❌ 工具调用失败:${e.message}`;
        }
    }

    /** 智能解析工具参数 */
    _parseToolParameters(toolName, parameters) {
        const paramDict = {};

        if (parameters.includes('=')) {
            // 格式: key=value 或 action=search,query=Python
            if (parameters.includes(',')) {
                // 多个参数:action=search,query=Python,limit=3
                const pairs = parameters.split(',');
                for (const pair of pairs) {
                    if (pair.includes('=')) {
                        const [key, value] = pair.split('=', 2);
                        paramDict[key.trim()] = value.trim();
                    }
                }
            } else {
                // 单个参数:key=value
                const [key, value] = parameters.split('=', 2);
                paramDict[key.trim()] = value.trim();
            }
        } else {
            // 直接传入参数，根据工具类型智能推断
            if (toolName === 'search') {
                paramDict.query = parameters;
            } else if (toolName === 'memory') {
                paramDict.action = 'search';
                paramDict.query = parameters;
            } else {
                paramDict.input = parameters;
            }
        }

        return paramDict;
    }

    /** 自定义的流式运行方法 */
    async *streamRun(inputText, kwargs = {}) {
        console.log(`🌊 ${this.name} 开始流式处理: ${inputText}`);

        const messages = [];

        if (this.systemPrompt) {
            messages.push({ role: 'system', content: this.systemPrompt });
        }

        for (const msg of this._history) {
            messages.push({ role: msg.role, content: msg.content });
        }

        messages.push({ role: 'user', content: inputText });

        // 流式调用 LLM
        let fullResponse = '';
        process.stdout.write('📝 实时响应: ');

        const stream = await this.llm.chatStream(messages);
        for await (const chunk of stream) {
            fullResponse += chunk;
            process.stdout.write(chunk);
            yield chunk;
        }

        console.log(); // 换行

        // 保存完整对话到历史记录
        this.addMessage(new Message('user', inputText));
        this.addMessage(new Message('assistant', fullResponse));
        console.log(`✅ ${this.name} 流式响应完成`);
    }

    /** 添加工具到 Agent（便利方法） */
    addTool(tool) {
        if (!this.toolRegistry) {
            // 需要动态导入 ToolRegistry
            // 注意：这里假设 ToolRegistry 在某个模块中导出
            // 实际使用时需要根据项目结构调整
            throw new Error('ToolRegistry 未初始化，请在构造函数中传入 toolRegistry 参数');
        }

        this.toolRegistry.registerTool(tool);
        this.enableToolCalling = true;
        console.log(`🔧 工具 '${tool.name}' 已添加`);
    }

    /** 检查是否有可用工具 */
    hasTools() {
        return this.enableToolCalling && this.toolRegistry !== null;
    }

    /** 移除工具（便利方法） */
    removeTool(toolName) {
        if (this.toolRegistry) {
            this.toolRegistry.unregister(toolName);
            return true;
        }
        return false;
    }

    /** 列出所有可用工具 */
    listTools() {
        if (this.toolRegistry) {
            return this.toolRegistry.listTools();
        }
        return [];
    }
}

export default SimpleAgent;
