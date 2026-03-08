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
- **排版规范来源：** 标点、空格、数学公式、代码块的写法，以上级目录的 `../文档编写规范-light-V1.1.1.md` 为准；如果与本文件其他条目冲突，以那份规范为准.
- **文件命名统一：** `note/` 下的文件名，除了章节编号前缀如 `4-3-` 保留连字符、以及术语本身已有连字符如 `Black-Scholes`、`Runge-Kutta` 这类情况外，其余分隔统一用空格，不额外用 `-` 充当标题分隔符.
- **不超纲：** 当前小节没引入的方法/名词，不要突然拓展或 name-drop. 如确实为了降低门槛需要补充，一律写进 `> **Note:** ...`，点到为止.
- **推导别跳步：** 关键的代数化简/近似展开尽量写出来，让读者能顺着每一步走到下一行.
- **结论别直给：** 像 "由 $|\epsilon_{k+1}|\approx L|\epsilon_k|^\alpha$ 推出 $\log|\epsilon_{k+1}|/\log|\epsilon_k|\to\alpha$" 这类用于数值判别收敛阶的公式，至少写出取对数、拆项/同除、以及为什么常数项在极限中消失.
- **直观解释先从最短版本写：** 如果一个概念的补充说明只需要帮助读者建立画面感，就优先用一句最短的类比或对照讲清楚，不要把 `Note` 扩成一大段，否则反而会冲淡主线、增加抽象感.

### 术语与记号

- **术语首次出现：** 用 `英文术语 (中文译名)`；如有常用缩写，可写成 `英文术语 (中文译名, 缩写)`，例如 `inverse quadratic interpolation (逆二次插值, IQI)`.
- **译名优先从中文数学习惯：** 术语翻译不要机械逐词对译，要优先采用教材或中文语境里更顺口、更常见的说法；例如这里用 "逆二次插值" 比 "逆向二次插值" 更自然，也更符合数值分析里常见的表达.
- **数学对象用 LaTeX：** 正文里提及函数、导数、误差、向量/矩阵等，尽量写成 $f$、$f'$、$\epsilon_k$、$\mathbf{x}$ 这种数学环境，避免纯文本 f / f' (代码块与函数参数名除外).

### Callout 引用块 (Definition / Theorem / Algorithm / ...)

教材里的 Definition / Theorem / Algorithm / Example / Demo / Observation / Lemma / Corollary / Function / Table 等，统一用英文，并整体包在引用块中.

- **短标题格式：** 如果冒号后面是一个简短标题，统一写成
  - `> **Definition:** **Fixed-point problem**.`
  - `> **Theorem:** **Contraction mapping**.`
  - 句点放在加粗外，避免写成 `**Term.**`
- **Block 标签统一用英文冒号：** `> **Note:**`、`> **Definition:**`、`> **Algorithm:**` 这类英文 callout 标签后，一律写英文冒号 `:`，不要写中文冒号 `：`.
- **标题后能顺接就顺接：** 如果 `Definition`、`Example`、`Algorithm`、`Function`、`Demo`、`Observation`、`Note` 这类 callout 标题后面只是接简短定义、说明、引导语或副标题，例如 `Given ...`、`Let ...`、`The steady state ...`，优先直接接在同一个引用块行里，不必为了版式单独换一行；如果后面连续几行其实还是同一段说明文字，也继续顺接，直到遇到公式、列表、代码块等需要另起一行的内容为止.
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
- **原文已经简洁时不要故意说复杂：** 如果教材本来只是在说 "线性" 与 "非线性" 这类直接对照，就按中文顺口的方式直说，不要为了显得更生动或更完整，自行扩成 "以任意方式进入模型" 这类更绕的表述.
- **正文里能不用符号缩写就不用：** 像 "$\sim R$ 的尺度" 这类需要读者额外停下来解释的缩写，如果直接改成 "不能忽略残差 $R$ 的影响"、"和 $R$ 差不多大" 之类的中文就能讲清楚，就优先用中文说法.
- **面向初学者时优先展开向量和矩阵：** 如果某个向量或 Jacobian 写成下标压缩记号会显得太抽象，而展开成列向量或矩阵更直观，那就优先写成展开形式，减少读者在记号之间来回翻译的负担.
- **区分函数和取值：** 像 $\mathbf{f}(\mathbf{x})$ 这类依赖参数的对象，在首次引入时优先称作 "残差函数"、"向量值函数"；只有在固定某个参数值之后，再说它的取值是 "残差向量".
- **初学者版本尽量少引入临时符号：** 如果教材和上下文已经一直在用 $\mathbf{s}_k$、$\mathbf{y}_k$、$\mathbf{A}_k$ 这类迭代记号，就继续沿用，不要为了形式上更抽象而额外引入 $\mathbf{s}$、$\mathbf{z}$ 这类临时变量；只有在确实不可避免时才引入新符号，并立刻说明它的角色.
- **新符号密集出现时先补一轮流程 Note：** 如果一段文字里同时引入 $\mathbf{s}_k$、$\mathbf{y}_k$、$\mathbf{A}_k$ 这类多个新符号，后面又马上进入更新条件或矩阵公式，那么先用一个短 `Note` 按 "先算什么、再算什么、最后更新什么" 的顺序，把一轮迭代讲清楚，再继续正式推导.
- **非平凡更新公式后补来源：** 像 Broyden 更新这类不容易一眼看出的公式，不要只给一个 `Definition` 就结束；紧跟一个短 `Note`，把 "先设成什么形式、代入什么条件、怎样整理出来" 写清楚，避免读者只看到结论.

### 规则沉淀

当用户在对话里提出新的笔记写作改进点：

1) 先用一句话总结成可执行规则.
2) 把它追加到本文件对应条目里.
3) 后续笔记严格执行.
