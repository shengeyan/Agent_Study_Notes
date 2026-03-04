/*
 * @lc app=leetcode.cn id=8 lang=javascript
 *
 * [8] 字符串转换整数 (atoi)
 */

// @lc code=start
/**
 * @param {string} s
 * @return {number}
 */
var myAtoi = function (s) {
    const INT_MAX = 2147483647; // 2^31 - 1
    const INT_MIN = -2147483648; // -2^31
    
    let submmit = 0;
    let f = 1;
    let hasSign = false; // 是否已经处理过符号
    let hasDigit = false; // 是否已经开始读取数字

    for (const i of s) {
        // 跳过前导空格（只在未开始读取数字和符号时）
        if (i === ' ' && !hasSign && !hasDigit) {
            continue;
        }

        // 处理符号（只能出现一次，且在数字之前）
        if ((i === '+' || i === '-') && !hasSign && !hasDigit) {
            hasSign = true;
            if (i === '-') {
                f = -1;
            }
            continue;
        }

        // 处理数字
        if (i >= '0' && i <= '9') {
            hasDigit = true;
            const digit = parseInt(i);
            
            // 检查溢出
            if (submmit > Math.floor(INT_MAX / 10) || 
                (submmit === Math.floor(INT_MAX / 10) && digit > 7)) {
                return f === 1 ? INT_MAX : INT_MIN;
            }
            
            submmit = submmit * 10 + digit;
        } else {
            // 遇到非数字字符，停止解析
            break;
        }
    }

    return f * submmit;
};
// @lc code=end
