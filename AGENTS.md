# AGENTS.md instructions for G:\其他计算机\我的 Mac\Google Drive\250321-PAN-Numerical analysis seminar\numerical-analysis-seminar-datahub

<INSTRUCTIONS>
## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.
### Available skills
- skill-creator: Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Codex's capabilities with specialized knowledge, workflows, or tool integrations. (file: C:/Users/PC/.codex/skills/.system/skill-creator/SKILL.md)
- skill-installer: Install Codex skills into $CODEX_HOME/skills from a curated list or a GitHub repo path. Use when a user asks to list installable skills, install a curated skill, or install a skill from another repo (including private repos). (file: C:/Users/PC/.codex/skills/.system/skill-installer/SKILL.md)
### How to use skills
- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.
</INSTRUCTIONS>

<environment_context>
  <cwd>G:\其他计算机\我的 Mac\Google Drive\250321-PAN-Numerical analysis seminar\numerical-analysis-seminar-datahub</cwd>
  <shell>powershell</shell>
  <current_date>2026-03-08</current_date>
  <timezone>Asia/Shanghai</timezone>
</environment_context>

---

## 仓库内笔记改稿规范 (适用于 `note/` 下所有 Markdown)

### 目标与边界

- **中文叙事优先：** 尽量用更符合中文阅读习惯的 "大白话" 把同一含义讲清楚，避免逐句直译英文叙事逻辑.
- **排版规范来源：** 标点、空格、数学公式、代码块的写法，以上级目录的 `../文档编写规范-light-V1.1.3.md` 为准；如果与本文件其他条目冲突，以那份规范为准.
- **文件命名统一：** `note/` 下的文件名，除了章节编号前缀如 `4-3-` 保留连字符、以及术语本身已有连字符如 `Black-Scholes`、`Runge-Kutta` 这类情况外，其余分隔统一用空格，不额外用 `-` 充当标题分隔符.
- **不超纲：** 当前小节没引入的方法/名词，不要突然拓展或 name-drop. 如确实为了降低门槛需要补充，一律写进 `> **Note:** ...`，点到为止.
- **推导别跳步：** 关键的代数化简/近似展开尽量写出来，让读者能顺着每一步走到下一行.
- **结论别直给：** 像 "由 $|\epsilon_{k+1}|\approx L|\epsilon_k|^\alpha$ 推出 $\log|\epsilon_{k+1}|/\log|\epsilon_k|\to\alpha$" 这类用于数值判别收敛阶的公式，至少写出取对数、拆项/同除、以及为什么常数项在极限中消失.
- **直观解释先从最短版本写：** 如果一个概念的补充说明只需要帮助读者建立画面感，就优先用一句最短的类比或对照讲清楚，不要把 `Note` 扩成一大段，否则反而会冲淡主线、增加抽象感.
- **校稿时按两三句连起来看：** 不要只看单句通不通顺，而要把相邻两三句放在一起检查主语是否稳定、逻辑是否自然、有没有为了“更生动”反而把原本简洁的意思说复杂.

### 术语与记号

- **术语首次出现：** 用 `英文术语 (中文译名)`；如有常用缩写，可写成 `英文术语 (中文译名, 缩写)`，例如 `inverse quadratic interpolation (逆二次插值, IQI)`.
- **译名优先从中文数学习惯：** 术语翻译不要机械逐词对译，要优先采用教材或中文语境里更顺口、更常见的说法；例如这里用 "逆二次插值" 比 "逆向二次插值" 更自然，也更符合数值分析里常见的表达.
- **数学对象用 LaTeX：** 正文里提及函数、导数、误差、向量/矩阵等，尽量写成 $f$、$f'$、$\epsilon_k$、$\mathbf{x}$ 这种数学环境，避免纯文本 f / f' (代码块与函数参数名除外).
- **LaTeX 公式先按“人怎么读”来排：** 公式不是写给编译器看的源码，而是写给读者看的数学句子；空格的作用是把并列项、运算、范围、条件这些结构自然地分开，让读者一眼看出“哪里该连着看，哪里该断开看”，而不是把整串符号重新拆词.
- **LaTeX 公式里的空格优先服务可读性：** 关系符、二元运算符、逗号分隔，以及 `\dots` / `\ldots` 这类省略记号附近，只要语义上是在连接并列对象或表达一个关系，就优先留出适当空格；例如写成 $t_0 < t_1 < \cdots < t_n$、$p(t_k) = y_k$、$y_0, \ldots, y_n$、$t_0, \dots, t_n$、$\mathbf{y}, \mathbf{z}$、$\alpha, \beta$、$n + 1$，不要为了“短”把它们全都挤在一起.
- **LaTeX 公式里的空格不要滥加：** 如果一个结构本来就是一个整体，例如下标、上标、函数名、命令名或紧密绑定的数学对象，就保持紧凑，不要硬塞空格；例如 $x_{k,m}$、$t_{k-1}$、$\phi_k$ 这类写法本来就足够清楚.

### Callout 引用块 (Definition / Theorem / Algorithm / ...)

教材里的 Definition / Theorem / Algorithm / Example / Demo / Observation / Lemma / Corollary / Function / Table 等，统一用英文，并整体包在引用块中.

- **短标题格式：** 如果冒号后面是一个简短标题，统一写成
  - `> **Definition:** **Fixed-point problem**.`
  - `> **Theorem:** **Contraction mapping**.`
  - 句点放在加粗外，避免写成 `**Term.**`
- **Block 标签统一用英文冒号：** `> **Note:**`、`> **Definition:**`、`> **Algorithm:**` 这类英文 callout 标签后，一律写英文冒号 `:`，不要写中文冒号 `：`.
- **标题后能顺接就顺接：** 如果 `Definition`、`Example`、`Algorithm`、`Function`、`Demo`、`Observation`、`Note`、`Proof` 这类 callout 标题后面只是接简短定义、说明、引导语、证明起手句或副标题，例如 `Given ...`、`Let ...`、`By linearity ...`、`The steady state ...`，优先直接接在同一个引用块行里，不必为了版式单独换一行；如果后面连续几行其实还是同一段说明文字，也继续顺接，直到遇到公式、列表、代码块等需要另起一行的内容为止.
- **有短标题的定理类也优先顺接：** 如果 `Theorem`、`Lemma`、`Corollary` 已经先给了一个短标题，那么标题后的正文也优先接在同一行里；只有在没有短标题、冒号后直接进入完整命题陈述时，才拆成两行.
- **定理类完整陈述仍拆两行：** 如果是 `Theorem`、`Lemma`、`Corollary` 一类需要完整陈述条件和结论的命题，且冒号后面直接进入完整陈述句 (往往以 If / For / Suppose / Consider ... 开头)，则拆成两行
  - `> **Theorem:**`
  - `> If ...`

### 中文叙事微调经验 (常见改法)

- **等价关系补一句：** 当出现 $A=B$、$f(r)=0\\Leftrightarrow g(r)=r$ 这类等价关系时，通常在公式后补一句 "也就是说，..."，把等价关系翻译成人话，读起来更顺.
- **把英文逻辑改成中文顺序：** 例如 "这是第一个即使精确算术也不会有限步到达答案的典型例子" 更适合写成 "在实际计算中，即使计算精确，它也不会在有限步内到达不动点；我们只能生成一个序列，觉得够近了就停."
- **抽象操作先换成图像对照：** 像 "把抛物线侧过来" 这种说法如果还显得悬，可以直接改写成 "拟合一条横着的抛物线 (形如 $x=y^2$)，而不是拟合一条竖着的抛物线 (形如 $y=x^2$)"；这种对照通常比继续解释定义更容易让读者一下抓住意思.
- **把 "$x$ 的函数" 改成 "关于 $x$ 的函数"：** 在中文数学表述里，像 Jacobian、导数、矩阵值函数这类对象，通常说 "它是关于 $\mathbf{x}$ 的函数" 更顺，不要写成 "它是 $\mathbf{x}$ 的函数".
- **不要硬贴英文叙事痕迹：** 像 "事后回看" 这类逐词翻译出来但不符合中文行文习惯的连接语，通常直接删掉，改成 "我们可以把……理解为……" 这类更自然的中文叙述.
- **句首英文术语注意大小写：** 如果英文术语出现在一句话开头，按正常英文书写习惯把首字母大写，例如写成 `Quasi-Newton methods`，不要保留句中小写形式.
- **把 "评估 $f$" 优先改成 "计算 $f$ 的数值"：** 在中文数值分析笔记里，描述 function evaluation 时，优先写成 "计算 $\mathbf{f}$ 的数值" 或 "函数值计算"，不要直接照搬 "评估 $\mathbf{f}$" 这种翻译腔说法.
- **能简就简，但别丢意思：** 如果一句话在不损失数学含义、也不引入歧义的前提下可以明显说得更短，就优先选更简洁的中文表述，例如把 "重复做这些函数值计算就显得浪费" 收成 "重复计算就比较浪费".
- **定义先说“它在说什么”，再说“它严格等价于什么”：** 遇到精度阶、收敛阶、条件数这类带形式定义的概念时，先用一句直白中文把读者真正该抓住的主意思说出来，例如 "误差主要像 $h^m$ 这样下降"；然后再补上严格表述，例如 "是 $O(h^m)$ 但不是 $O(h^{m+1})$". 这样读者先得到画面和判断标准，再理解技术边界，不会一上来被符号定义卡住.
- **修饰信息优先收进标题或括号：** 如果一句话里只是补充 "最自然但也危险"、"见前一节" 这类修饰信息，优先把它收进标题括号或句末括号；标题已经承载的信息，正文主句里就不要再重复一遍.
- **后文概述尽量落到具体方法：** 如果上下文已经明确后面会用哪类具体方法，就直接说 "分段低次多项式"、"分段线性插值" 这类更落地的对象，不要退回到 "每一段次数较小" 这种更虚的概括.
- **直觉型 Example 后补作用说明：** 如果一个 Example 主要是为了帮读者建立定义的边界感或直觉，而不是为了推出后续计算结果，就在后面补一个短 `Note`，直接说明 "这个例子为什么放在这里".
- **解释作用时少写元话语：** 如果 `Note` 只是解释一个 Example / Demo 的作用，优先直接写 "这个例子说明了什么"，不要写 "这个 Example 放在这里是为了……" 这类 AI 味较重的元表述.
- **原文已经简洁时不要故意说复杂：** 如果教材本来只是在说 "线性" 与 "非线性" 这类直接对照，就按中文顺口的方式直说，不要为了显得更生动或更完整，自行扩成 "以任意方式进入模型" 这类更绕的表述.
- **正文里能不用符号缩写就不用：** 像 "$\sim R$ 的尺度" 这类需要读者额外停下来解释的缩写，如果直接改成 "不能忽略残差 $R$ 的影响"、"和 $R$ 差不多大" 之类的中文就能讲清楚，就优先用中文说法.
- **面向初学者时优先展开向量和矩阵：** 如果某个向量或 Jacobian 写成下标压缩记号会显得太抽象，而展开成列向量或矩阵更直观，那就优先写成展开形式，减少读者在记号之间来回翻译的负担.
- **首次引入向量时优先直接写成列向量：** 如果一个向量第一次出现时，直接写成竖着的列向量比写成横排再加 $^{T}$ 更清楚，就优先直接展开，不为了版面紧凑牺牲可读性.
- **区分函数和取值：** 像 $\mathbf{f}(\mathbf{x})$ 这类依赖参数的对象，在首次引入时优先称作 "残差函数"、"向量值函数"；只有在固定某个参数值之后，再说它的取值是 "残差向量".
- **初学者版本尽量少引入临时符号：** 如果教材和上下文已经一直在用 $\mathbf{s}_k$、$\mathbf{y}_k$、$\mathbf{A}_k$ 这类迭代记号，就继续沿用，不要为了形式上更抽象而额外引入 $\mathbf{s}$、$\mathbf{z}$ 这类临时变量；只有在确实不可避免时才引入新符号，并立刻说明它的角色.
- **新符号密集出现时先补一轮流程 Note：** 如果一段文字里同时引入 $\mathbf{s}_k$、$\mathbf{y}_k$、$\mathbf{A}_k$ 这类多个新符号，后面又马上进入更新条件或矩阵公式，那么先用一个短 `Note` 按 "先算什么、再算什么、最后更新什么" 的顺序，把一轮迭代讲清楚，再继续正式推导.
- **非平凡更新公式后补来源：** 像 Broyden 更新这类不容易一眼看出的公式，不要只给一个 `Definition` 就结束；紧跟一个短 `Note`，把 "先设成什么形式、代入什么条件、怎样整理出来" 写清楚，避免读者只看到结论.

### 规则沉淀

当用户在对话里提出新的笔记写作改进点：

1) 先用一句话总结成可执行规则.
2) 把它追加到本文件对应条目里.
3) 后续笔记严格执行.
