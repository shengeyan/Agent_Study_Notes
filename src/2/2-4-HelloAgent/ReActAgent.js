import Agent from './Agent.js';
import Message from './Message.js';

const MY_REACT_PROMPT = `
你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- \`{{tool_name}}[{{tool_input}}]\` - 调用指定工具
- \`Finish[最终答案]\` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
`;

class ReActAgent extends Agent {
    constructor(name, llm, toolRegistry, systemPrompt = null, config = null, maxSteps = 5, customPrompt = null) {
        super(name, llm, systemPrompt, config);
        this.toolRegistry = toolRegistry;
        this.maxSteps = maxSteps;
        this.currentHistory = [];
        this.promptTemplate = customPrompt || MY_REACT_PROMPT;
        console.log(`✅ ${name} 初始化完成，最大步数: ${maxSteps}`);
    }

    /** 运行 ReAct Agent */
    async run(inputText, kwargs = {}) {
        this.currentHistory = [];
        let currentStep = 0;

        console.log(`\n🤖 ${this.name} 开始处理问题: ${inputText}`);

        while (currentStep < this.maxSteps) {
            currentStep++;
            console.log(`\n--- 第 ${currentStep} 步 ---`);

            // 1. 构建提示词
            const toolsDesc = this.toolRegistry.getToolsDescription();
            const historyStr = this.currentHistory.join('\n');
            const prompt = this.promptTemplate
                .replace('{tools}', toolsDesc)
                .replace('{question}', inputText)
                .replace('{history}', historyStr);

            // 2. 调用 LLM
            const messages = [{ role: 'user', content: prompt }];
            const responseText = await this.llm.chat(messages, kwargs);

            // 3. 解析输出
            const { thought, action } = this._parseOutput(responseText);

            // 4. 检查完成条件
            if (action && action.startsWith('Finish')) {
                const finalAnswer = this._parseActionInput(action);
                this._saveToHistory(inputText, finalAnswer);
                return finalAnswer;
            }

            // 5. 执行工具调用
            if (action) {
                const { toolName, toolInput } = this._parseAction(action);
                const observation = this.toolRegistry.executeTool(toolName, toolInput);
                this.currentHistory.push(`Action: ${action}`);
                this.currentHistory.push(`Observation: ${observation}`);
            }
        }

        // 达到最大步数
        const finalAnswer = '抱歉，我无法在限定步数内完成这个任务。';
        this._saveToHistory(inputText, finalAnswer);
        return finalAnswer;
    }

    /** 解析 LLM 输出，提取 Thought 和 Action */
    _parseOutput(text) {
        const thoughtMatch = text.match(/Thought:\s*(.+?)(?=\nAction:|$)/s);
        const actionMatch = text.match(/Action:\s*([\s\S]+?)(?=\nThought:|$)/);

        const thought = thoughtMatch ? thoughtMatch[1].trim() : null;
        const action = actionMatch ? actionMatch[1].trim() : null;

        if (thought) {
            console.log(`💭 思考: ${thought}`);
        }
        if (action) {
            console.log(`⚡ 行动: ${action}`);
        }

        return { thought, action };
    }

    /** 解析 Action 字符串，提取工具名和输入 */
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

    /** 解析 Finish 动作的输入 */
    _parseActionInput(action) {
        const match = action.match(/Finish\[([\s\S]*)\]/);
        return match ? match[1].trim() : action;
    }

    /** 保存对话到历史记录 */
    _saveToHistory(inputText, response) {
        this.addMessage(new Message('user', inputText));
        this.addMessage(new Message('assistant', response));
        console.log(`✅ ${this.name} 响应完成`);
    }
}

export default ReActAgent;
