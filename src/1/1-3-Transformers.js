import { AutoModelForCausalLM, AutoTokenizer } from '@xenova/transformers';

const model_id = 'Xenova/Qwen1.5-0.5B-Chat';

console.log('开始加载分词器...');

// 加载分词器
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
console.log('✓ 分词器加载完成');

// 加载模型
const model = await AutoModelForCausalLM.from_pretrained(model_id);
console.log('✓ 模型加载完成');

console.log('模型和分词器加载完成！');

// 准备对话输入
const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: '你好，请介绍你自己。' },
];

// 使用分词器的模板格式化输入
const text = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
});

// 编码输入文本
const model_inputs = tokenizer([text], { return_tensors: 'pt' });

console.log('编码后的输入文本:');
console.log(model_inputs);

console.log('\n=== 开始生成文本 ===');
// 使用模型生成回答
// max_new_tokens 控制了模型最多能生成多少个新的Token
console.log('正在生成（可能需要 10-30 秒）...');
const generated_ids = await model.generate(model_inputs.input_ids, {
    max_new_tokens: 512,
});
console.log('✓ 生成完成');

// 将生成的 Token ID 截取掉输入部分
// 这样我们只解码模型新生成的部分
const output_ids = generated_ids[0].slice(model_inputs.input_ids[0].length);

// 解码生成的 Token ID
const response = tokenizer.decode(output_ids, { skip_special_tokens: true });

console.log('\n模型的回答:');
console.log(response);
