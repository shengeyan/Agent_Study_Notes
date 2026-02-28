const VALID_ROLES = ['user', 'system', 'assistant', 'tool'];

class Message {
    constructor(role, content, options = {}) {
        if (!VALID_ROLES.includes(role)) {
            throw new Error(`Invalid role: "${role}". Must be one of: ${VALID_ROLES.join(', ')}`);
        }

        this.role = role;
        this.content = content;
        this.timestamp = options.timestamp || new Date();
        this.metadata = options.metadata || {};
    }

    toDict() {
        return {
            role: this.role,
            content: this.content,
        };
    }

    toString() {
        return `[${this.role}] ${this.content}`;
    }
}

export default Message;
export { VALID_ROLES };
