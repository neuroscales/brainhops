# CLAUDE.md — brainhops

Guidance for coding agents and contributors working in this repository.
The organisation-wide rules in `bagofseeds/.github/CLAUDE.md` (workflow,
gate before every PR, code conventions) apply here as well. This file
records the rules for documentation, which are stricter than the
organisation-wide ones.

## Documentation

Documentation here means the README, the pages under `docs/`, the design
memos in `docs/design/`, the specification notes in `spec/`, every
docstring and every explanatory comment in `src/` and `tests/`, and the
`tx.Doc(...)` strings that annotate attributes. All of it follows the
rules below.

### Writing style

- Write complete sentences with an explicit subject and verb. Keep
  articles and prepositions. Do not write fragments, and do not use
  symbol shorthand (arrows, ampersands, "w/", "vs") in prose.
- Be concise by removing content that does not help the reader, never by
  compressing grammar. A longer sentence that reads naturally is better
  than a short one that has to be decoded. Existing text is a source of
  facts, not of style: do not imitate it if it is awkward.
- Start with what the object is for and when someone meets it, in
  everyday words. Then explain how it works, then the details and the
  edge cases. A call signature or a calling convention is never the first
  sentence.
- Write for a contributor who has not read the module. Use technical
  terms where they name a precise concept, define a term in a short
  clause the first time it appears, and link it to where it is explained.
  Do not use internal jargon (for example "kind node", "is keyed by", "is
  lowered to") when an ordinary verb or phrase says the same thing.
- Repeat a noun instead of chaining "it", "this" and "that" when the
  reference could be unclear. Prefer common verbs to jargon verbs.
- Use em dashes, colons, parentheses and semicolons sparingly. Do not
  use bold text or "Note:" callouts. Use a list only for content that is
  genuinely a list.
- Every statement must be checked against the code. When the old text and
  the code disagree, document what the code does and open an issue about
  the discrepancy.

As an example, this is rejected:

> Tier 1: same type on the same orientation line, the strongest signal.
> It pairs right-to-left with left-to-right whatever the names.

and this is wanted:

> First tier: axes of the same type that lie along the same oriented
> line. This is the strongest signal, and it pairs a right-to-left axis
> with a left-to-right one regardless of their names.

### Docstring format

- Docstrings use the NumPy style, rendered by mkdocstrings. The first
  line is a summary in the imperative mood ("Return ...") for functions
  and a noun phrase for classes and properties.
- In the `Parameters`, `Other Parameters`, `Attributes` and `Receives`
  sections, each variable has its own entry. Never group names
  (`a, b : int` is wrong).
- Every entry has a type: `name : type`. This includes `*args` and
  `**kwargs`.
- A parameter that has a default in the signature carries a flag after the
  type: `name : int, default=3` when the default is a value other than
  `None`, and `name : int, optional` when the default is `None` or a
  sentinel. A required parameter has no flag. The names and the order
  must match the signature.
- `Returns` and `Yields` list one entry per returned item. `Raises` lists
  the exception class only.
- Admonitions (`!!! note`, `!!! example`, `!!! warning`) go before the
  `Parameters` section, because mkdocstrings would otherwise merge them
  into the generated table.
- Cross-references use the short form (`` [`Name`][] ``) when the symbol
  is defined or imported in the module, and the long form
  (`` [`Name`][package.module.Name] ``) otherwise. The text in the left
  brackets is always in backticks.
- Examples in `pycon` blocks must run on Python 3.8: no `Annotated`, no
  `list[int]`, no `X | Y`.
- Every enum class has, in its docstring, a table of all its members with
  their values, so that the values appear on the documentation site. RST
  grid tables and Markdown pipe tables both render.

### Line length

Code is limited to 79 columns by `ruff`. Docstrings and comments should
wrap at 72 columns when possible, as recommended by PEP 8. The exceptions
are URLs, long cross-reference links, tables, ASCII drawings and
doctest output, which may use the full 79 columns or more. `ruff` does not
enforce the 72-column rule, so it is a convention to follow when writing
or editing text.

### What must be kept

A rewrite of documentation must not lose information that is not prose.
Keep:

- section separators and banners in every style (`# --- title ----`,
  `# -- title ---`, framed blocks made of `-` or `=` rules). Restore them
  if an edit removes them;
- ASCII art and diagrams, byte for byte;
- links, DOIs, and issue and pull request references (`#123`) with the
  sentence that cites them;
- tables, code examples and command lines, which are data and are copied
  exactly;
- directive comments (`# noqa`, `# type:`, `# pragma`, `# fmt:`,
  `# ruff:`) and `TODO` / `FIXME` notes.

### Documentation-only changes

- A change that only touches documentation must not change any code.
  Check it by comparing the syntax trees of the old and the new file with
  docstrings and `tx.Doc(...)` strings removed.
- Run `ruff check .` and `ruff format --check .` from the repository
  root (they also check Python code blocks in the Markdown pages), and
  `codespell -I codespell-ignore-words.txt` on the changed files.
- Pages under `docs/api/` use `::: package.module` directives. A
  directive must name a module that exists.
- Design memos in `docs/design/` record the decision, the reason for it
  and its status (implemented or not). They do not keep the history of
  how the decision was reached.
