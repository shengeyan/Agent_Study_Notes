/*
 * @lc app=leetcode.cn id=2 lang=javascript
 *
 * [2] 两数相加
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var addTwoNumbers = function (l1, l2) {
    let head = new ListNode(0);
    let current = head;  // 用于尾插法
    let front = 0;

    while (l1 || l2) {
        let one = l1 ? l1.val : 0;
        let two = l2 ? l2.val : 0;

        let sum = one + two + front;
        front = Math.floor(sum / 10);
        sum = sum % 10;

        current.next = new ListNode(sum);  // 尾插法
        current = current.next;

        l1 && (l1 = l1.next);
        l2 && (l2 = l2.next);
    }

    // 处理最高位进位
    if (front > 0) {
        current.next = new ListNode(front);
    }

    return head.next;  // 返回真正的头节点
};
// @lc code=end
