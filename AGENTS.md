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
- **不超纲：** 当前小节没引入的方法/名词，不要突然拓展或 name-drop. 如确实为了降低门槛需要补充，一律写进 `> **Note:** ...`，点到为止.
- **推导别跳步：** 关键的代数化简/近似展开尽量写出来，让读者能顺着每一步走到下一行.

### 术语与记号
- **术语首次出现：** 用 `英文术语 (中文译名)`；如有常用缩写，可写成 `英文术语 (中文译名, 缩写)`，例如 `inverse quadratic interpolation (逆向二次插值, IQI)`.
- **数学对象用 LaTeX：** 正文里提及函数、导数、误差、向量/矩阵等，尽量写成 $f$、$f'$、$\epsilon_k$、$\mathbf{x}$ 这种数学环境，避免纯文本 f / f' (代码块与函数参数名除外).

### Callout 引用块 (Definition / Theorem / Algorithm / ...)
教材里的 Definition / Theorem / Algorithm / Example / Demo / Observation / Lemma / Corollary / Function / Table 等，统一用英文，并整体包在引用块中.

- **短标题格式：** 如果冒号后面是一个简短标题，统一写成
  - `> **Definition:** **Fixed-point problem**.`
  - `> **Theorem:** **Contraction mapping**.`
  - 句点放在加粗外，避免写成 `**Term.**`
- **完整陈述句：** 如果冒号后面直接就是陈述句 (往往以 If / For / Let / Suppose / Consider ... 开头)，则拆成两行
  - `> **Theorem:**`
  - `> If ...`

### 中文叙事微调经验 (常见改法)
- **等价关系补一句：** 当出现 $A=B$、$f(r)=0\\Leftrightarrow g(r)=r$ 这类等价关系时，通常在公式后补一句 "也就是说，..."，把等价关系翻译成人话，读起来更顺.
- **把英文逻辑改成中文顺序：** 例如 "这是第一个即使精确算术也不会有限步到达答案的典型例子" 更适合写成 "在实际计算中，即使计算精确，它也不会在有限步内到达不动点；我们只能生成一个序列，觉得够近了就停."

### 规则沉淀
当用户在对话里提出新的笔记写作改进点：
1) 先用一句话总结成可执行规则.
2) 把它追加到本文件对应条目里.
3) 后续笔记严格执行.

