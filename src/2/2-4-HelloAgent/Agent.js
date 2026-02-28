import Message from './Message.js';
import MyLLM from './MyLLM.js';
import Config from './Config.js';

/**
 * Agent 基类（抽象类）
 */
class Agent {
    constructor(name, llm, systemPrompt = null, config = null) {
        // 检查是否直接实例化抽象类
        if (new.target === Agent) {
            throw new TypeError('Cannot instantiate abstract class Agent directly');
        }

        this.name = name;
        this.llm = llm;
        this.systemPrompt = systemPrompt;
        this.config = config || new Config();
        this._history = [];
    }

    /**
     * 运行 Agent（抽象方法，子类必须实现）
     * @param {string} inputText - 用户输入
     * @param {Object} kwargs - 额外参数
     * @returns {string}
     */
    run(inputText, kwargs = {}) {
        throw new Error('Method run() must be implemented by subclass');
    }

    /**
     * 添加消息到历史记录
     * @param {Message} message
     */
    addMessage(message) {
        this._history.push(message);
    }

    /**
     * 清空历史记录
     */
    clearHistory() {
        this._history = [];
    }

    /**
     * 获取历史记录（返回副本）
     * @returns {Message[]}
     */
    getHistory() {
        return [...this._history]; // 返回副本，避免外部修改
    }

    /**
     * 字符串表示
     * @returns {string}
     */
    toString() {
        return `Agent(name=${this.name}, provider=${this.llm.provider})`;
    }
}

export default Agent;
