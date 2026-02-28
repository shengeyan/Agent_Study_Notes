import dotenv from 'dotenv';

// 加载环境变量
dotenv.config();

/**
 * HelloAgents 配置类
 */
class Config {
    constructor(options = {}) {
        // LLM 配置
        this.defaultModel = options.defaultModel || 'gpt-3.5-turbo';
        this.defaultProvider = options.defaultProvider || 'openai';
        this.temperature = options.temperature !== undefined ? options.temperature : 0.7;
        this.maxTokens = options.maxTokens || null;
        
        // 系统配置
        this.debug = options.debug !== undefined ? options.debug : false;
        this.logLevel = options.logLevel || 'INFO';
        
        // 其他配置
        this.maxHistoryLength = options.maxHistoryLength || 100;
    }

    /**
     * 从环境变量创建配置
     * @returns {Config}
     */
    static fromEnv() {
        return new Config({
            debug: process.env.DEBUG?.toLowerCase() === 'true',
            logLevel: process.env.LOG_LEVEL || 'INFO',
            temperature: process.env.TEMPERATURE ? parseFloat(process.env.TEMPERATURE) : 0.7,
            maxTokens: process.env.MAX_TOKENS ? parseInt(process.env.MAX_TOKENS) : null,
        });
    }

    /**
     * 转换为普通对象
     * @returns {Object}
     */
    toDict() {
        return {
            defaultModel: this.defaultModel,
            defaultProvider: this.defaultProvider,
            temperature: this.temperature,
            maxTokens: this.maxTokens,
            debug: this.debug,
            logLevel: this.logLevel,
            maxHistoryLength: this.maxHistoryLength,
        };
    }
}

export default Config;
