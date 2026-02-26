/**
 * Agent History
 * 符号假说：专家系统
 * ELIZA设计思想:关键词识别与排序、分解、重组、代词转换
 */

import readline from 'readline';

// 定义规则库
const rules = {
    'I need (.*)': ['Why do you need {0}?', 'Would it really help you to get {0}?'],
    "Why don't you (.*?)\\?": ["Do you really think I don't {0}?", 'Perhaps eventually I will {0}.'],
    'I am (.*)': ['Did you come to me because you are {0}?', 'How long have you been {0}?'],
    '.* mother .*': ['Tell me more about your mother.'],
    '.*': ['Please tell me more.', 'Can you elaborate on that?'],
};

// 定义代词转换
const pronounSwap = {
    i: 'you',
    you: 'i',
    me: 'you',
    my: 'your',
    am: 'are',
    mine: 'yours',
};

// 替换代词
function swapPronouns(phrase) {
    const words = phrase.toLowerCase().split(' ');
    const swapped = words.map(word => pronounSwap[word] || word);
    return swapped.join(' ');
}

// 响应生成函数
function respond(userInput) {
    // 遍历所有规则
    for (const [pattern, responses] of Object.entries(rules)) {
        const regex = new RegExp(pattern, 'i'); // 'i' 表示忽略大小写
        const match = userInput.match(regex);

        if (match) {
            // 提取捕获组（如果有）
            const captured = match[1] || '';

            // 代词转换
            const swapped = swapPronouns(captured);

            // 随机选择响应模板
            const template = responses[Math.floor(Math.random() * responses.length)];

            // 替换占位符
            const response = template.replace('{0}', swapped);

            return response;
        }
    }

    // 默认响应
    return 'Please tell me more.';
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

console.log('Therapist: Hello! How can I help you today?');

rl.on('line', input => {
    if (['quit', 'exit', 'bye'].includes(input.toLowerCase())) {
        console.log('Therapist: Goodbye. It was nice talking to you.');
        rl.close();
        return;
    }

    const response = respond(input);
    console.log(`Therapist: ${response}`);
    rl.prompt();
});

rl.prompt();
