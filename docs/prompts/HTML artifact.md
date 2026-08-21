请基于当前 diff / spec,生成一个自包含 HTML 文件,保存到
runtime/html-artifacts/<date>-pr-review.html。

要求:
- 单文件,CSS/JS 内联,可离线打开
- 顶部:审查结论(通过 / 有条件通过 / 需修改)
- 按严重程度分组:blocking / question / nit
- 每个 finding 附:文件路径、行号、原因、建议
- 附验证证据:跑了哪些测试、哪些没跑
- 不相关文件默认折叠,重点文件展开
- 不要平铺全部 diff,只展示需要人看的 20%