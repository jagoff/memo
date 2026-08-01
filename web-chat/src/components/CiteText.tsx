import { Children, Fragment, cloneElement, isValidElement, useCallback, type ReactNode } from "react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

const CITE_RE = /\[([a-f0-9]{6,32})\]/gi;
const BARE_LOCATOR_RE =
  /^repo:([^:\s]+):([^\n[\]]+?\.[a-z]{1,5})(?::(\d+-\d+))?@[0-9a-f]{6,40}$/i;
const HUMAN_VAULT_PATH_RE =
  /^((?:obsidian-[^:\s]+|Notes)):\s+([^\n[\]]+?\.md(?::\d+-\d+)?)$/i;
const OBSIDIAN_URL_RE = /^obsidian:\/\//i;

type InlinePart =
  | string
  | { kind: "cite"; id: string; n: number }
  | { kind: "wikilink"; target: string; label: string }
  | { kind: "rawUrl"; href: string }
  | { kind: "locator"; vault: string; path: string; lines: string }
  | { kind: "vaultPath"; vault: string; path: string; label: string }
  | { kind: "obsidianUri"; href: string; label: string };

// Path may be "file.md:81-148" — strip line range for the obsidian:// URL.
function stripLineRangeLocal(path: string): string {
  const idx = path.lastIndexOf(":");
  if (idx <= 0) return path;
  const tail = path.slice(idx + 1);
  if (/^\d+(-\d+)?$/.test(tail)) return path.slice(0, idx);
  return path;
}

function buildObsidianOpen(vault: string, path: string): string {
  return (
    "obsidian://open?vault=" +
    encodeURIComponent(vault) +
    "&file=" +
    encodeURIComponent(stripLineRangeLocal(path))
  );
}

function basenameOf(path: string): string {
  const clean = stripLineRangeLocal(path);
  const i = clean.lastIndexOf("/");
  return i >= 0 ? clean.slice(i + 1) : clean;
}

function obsidianUriLabel(uri: string): string {
  try {
    const u = new URL(uri);
    const file = u.searchParams.get("file") || u.searchParams.get("query") || "";
    if (file) return basenameOf(file);
  } catch {
    // ignore
  }
  return "obsidian link";
}

function chatUrlTransform(url: string): string {
  const trimmed = url.trim();
  if (OBSIDIAN_URL_RE.test(trimmed)) return trimmed;
  return defaultUrlTransform(url);
}

function renderInline(text: string, citeNumbers: Map<string, number>): InlinePart[] {
  const parts: InlinePart[] = [];
  // Wikilinks → obsidian://… → bare repo locators → bare URLs → [hash] cites.
  // Inline bold/italic/code/links are handled natively by react-markdown.
  const tokenRe =
    /(\[\[[^\]\n]+\]\]|obsidian:\/\/[^\s<>"'`)]+|https?:\/\/[^\s<>"'`)]+|repo:[^:\s]+:[^\n[\]]+?\.[a-z]{1,5}(?::\d+-\d+)?@[0-9a-f]{6,40}\b|(?:obsidian-[^:\s]+|Notes):\s+[^\n[\]]+?\.md(?::\d+-\d+)?|\[[a-f0-9]{6,32}\])/gi;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = tokenRe.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("[[") && tok.endsWith("]]")) {
      const inner = tok.slice(2, -2);
      const [target, alias] = inner.includes("|") ? inner.split("|", 2) : [inner, inner];
      parts.push({ kind: "wikilink", target: target.trim(), label: alias.trim() });
    } else if (tok.startsWith("obsidian://")) {
      let href = tok;
      while (href.length > 11 && /[.,;:!?)\]]$/.test(href)) {
        href = href.slice(0, -1);
      }
      parts.push({ kind: "obsidianUri", href, label: obsidianUriLabel(href) });
      last = m.index + href.length;
      continue;
    } else if (tok.startsWith("http://") || tok.startsWith("https://")) {
      let href = tok;
      while (href.length > 8 && /[.,;:!?)\]]$/.test(href)) {
        href = href.slice(0, -1);
      }
      parts.push({ kind: "rawUrl", href });
      last = m.index + href.length;
      continue;
    } else if (tok.startsWith("repo:")) {
      const lm = tok.match(BARE_LOCATOR_RE);
      if (lm) {
        const [, vault, rawPath, lines] = lm;
        parts.push({
          kind: "locator",
          vault: vault.trim(),
          path: rawPath.trim(),
          lines: (lines || "").trim(),
        });
      } else {
        parts.push(tok);
      }
    } else if (/^(?:obsidian-[^:\s]+|Notes):/i.test(tok)) {
      const vm = tok.match(HUMAN_VAULT_PATH_RE);
      if (vm) {
        const [, vault, rawPath] = vm;
        parts.push({
          kind: "vaultPath",
          vault: vault.trim(),
          path: rawPath.trim(),
          label: tok,
        });
      } else {
        parts.push(tok);
      }
    } else if (tok.startsWith("[")) {
      const id = tok.slice(1, -1).toLowerCase();
      const n = citeNumbers.get(id);
      if (n !== undefined) parts.push({ kind: "cite", id, n });
      else parts.push(tok);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function obsidianSearchUrl(target: string): string {
  return `obsidian://search?query=${encodeURIComponent(target)}`;
}

function sectionToneFor(headerText: string): string {
  const t = headerText.toLowerCase();
  if (t.includes("vault")) return "section-vault";
  if (t.includes("memflow")) return "section-memflow";
  if (t.includes("memo")) return "section-memo";
  if (t.includes("conflicto") || t.includes("conflict")) return "section-conflict";
  return "section-default";
}

function makeProcessor(
  citeNumbers: Map<string, number>,
  onCiteClick: (n: number) => void,
) {
  // Recursively walk react-markdown's children: strings get inline-tokenized
  // for citations / wikilinks / bare URLs; elements keep their type but have
  // their own children walked too.
  const processString = (text: string, keyBase: string): ReactNode[] => {
    return renderInline(text, citeNumbers).map((p, i) => {
      const key = `${keyBase}-${i}`;
      if (typeof p === "string") return <Fragment key={key}>{p}</Fragment>;
      if (p.kind === "cite") {
        return (
          <sup
            key={key}
            className="cite"
            onClick={() => onCiteClick(p.n)}
            title={p.id}
          >
            [{p.n}]
          </sup>
        );
      }
      if (p.kind === "wikilink") {
        return (
          <a
            key={key}
            className="md-wikilink"
            href={obsidianSearchUrl(p.target)}
            title={`Buscar "${p.target}" en Obsidian`}
          >
            {p.label}
          </a>
        );
      }
      if (p.kind === "locator") {
        const href = buildObsidianOpen(p.vault, p.path);
        const base = basenameOf(p.path);
        const label = p.lines ? `${base} (L${p.lines})` : base;
        return (
          <a
            key={key}
            className="md-wikilink"
            href={href}
            title={`Abrir ${p.path} en Obsidian`}
          >
            {label}
          </a>
        );
      }
      if (p.kind === "vaultPath") {
        return (
          <a
            key={key}
            className="md-wikilink"
            href={buildObsidianOpen(p.vault, p.path)}
            title={`Abrir ${p.path} en Obsidian`}
          >
            {p.label}
          </a>
        );
      }
      if (p.kind === "obsidianUri") {
        return (
          <a
            key={key}
            className="md-wikilink"
            href={p.href}
            title={p.href}
          >
            {p.label}
          </a>
        );
      }
      // rawUrl
      return (
        <a
          key={key}
          className="md-url"
          href={p.href}
          target="_blank"
          rel="noopener noreferrer"
          title={p.href}
        >
          {p.href}
        </a>
      );
    });
  };

  const walk = (children: ReactNode, keyBase = "w"): ReactNode => {
    const out: ReactNode[] = [];
    Children.toArray(children).forEach((child, idx) => {
      if (typeof child === "string") {
        out.push(...processString(child, `${keyBase}-${idx}`));
        return;
      }
      if (typeof child === "number" || typeof child === "boolean") {
        out.push(child);
        return;
      }
      if (isValidElement<{ children?: ReactNode }>(child)) {
        const grand = child.props?.children;
        if (grand === undefined) {
          out.push(child);
          return;
        }
        const newChildren = walk(grand, `${keyBase}-${idx}`);
        out.push(cloneElement(child, undefined, newChildren));
        return;
      }
      out.push(child);
    });
    return out;
  };

  return { walk, processString };
}

function asPlainText(children: ReactNode): string {
  let out = "";
  Children.toArray(children).forEach((c) => {
    if (typeof c === "string" || typeof c === "number") {
      out += String(c);
    } else if (isValidElement<{ children?: ReactNode }>(c)) {
      out += asPlainText(c.props?.children);
    }
  });
  return out;
}

// Returns the inner text when the paragraph's content is exactly one <strong>
// with at most a trailing ":" / whitespace after it. Otherwise null.
function detectSectionHeader(children: ReactNode): string | null {
  const arr = Children.toArray(children).filter((c) => !(typeof c === "string" && c.trim() === ""));
  if (arr.length === 0 || arr.length > 2) return null;
  const first = arr[0];
  if (!isValidElement<{ children?: ReactNode }>(first)) return null;
  if (first.type !== "strong") return null;
  const inner = asPlainText(first.props?.children).trim();
  if (!inner) return null;
  if (arr.length === 1) return inner;
  const tail = arr[1];
  if (typeof tail !== "string") return null;
  if (tail.trim() !== ":") return null;
  return `${inner}:`;
}

export function CiteText({
  text,
  citeNumbers,
}: {
  text: string;
  citeNumbers: Map<string, number>;
}) {
  const handleClick = useCallback((n: number) => {
    const el = document.getElementById(`source-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
    const parent = el.closest("details");
    if (parent && !parent.open) parent.open = true;
  }, []);

  const { walk } = makeProcessor(citeNumbers, handleClick);

  const components: Components = {
    p({ children }) {
      // Preserve the legacy "section header" visual: a paragraph whose entire
      // content is a single <strong> (optionally followed by a trailing ":")
      // renders as a colored section heading rather than a bold paragraph.
      const header = detectSectionHeader(children);
      if (header) {
        return (
          <h3 className={`md-section ${sectionToneFor(header)}`}>
            {header.replace(/:\s*$/, "")}
          </h3>
        );
      }
      return <p className="md-p">{walk(children)}</p>;
    },
    h1({ children }) {
      return <h2 className="md-h2">{walk(children)}</h2>;
    },
    h2({ children }) {
      return <h2 className="md-h2">{walk(children)}</h2>;
    },
    h3({ children }) {
      return <h3 className="md-h3">{walk(children)}</h3>;
    },
    h4({ children }) {
      return <h4 className="md-h4">{walk(children)}</h4>;
    },
    h5({ children }) {
      return <h4 className="md-h4">{walk(children)}</h4>;
    },
    h6({ children }) {
      return <h4 className="md-h4">{walk(children)}</h4>;
    },
    ul({ children }) {
      return <ul className="md-list">{walk(children)}</ul>;
    },
    ol({ children }) {
      return <ol className="md-list md-list-ordered">{walk(children)}</ol>;
    },
    li({ children }) {
      return <li className="md-li">{walk(children)}</li>;
    },
    blockquote({ children }) {
      return <blockquote className="md-quote">{walk(children)}</blockquote>;
    },
    strong({ children }) {
      return <strong className="md-bold">{walk(children)}</strong>;
    },
    em({ children }) {
      return <em className="md-italic">{walk(children)}</em>;
    },
    a({ href, children }) {
      const isObsidian = !!href && OBSIDIAN_URL_RE.test(href);
      return (
        <a
          className={isObsidian ? "md-wikilink" : "md-url"}
          href={href ?? "#"}
          target="_blank"
          rel="noopener noreferrer"
          title={href}
        >
          {isObsidian ? children : walk(children)}
        </a>
      );
    },
    code({ className, children }) {
      // Fenced code blocks come wrapped in <pre><code>; react-markdown sets a
      // `language-*` className. Inline code has no className.
      const isBlock = !!(className && className.startsWith("language-"));
      if (isBlock) {
        return <code className={`md-code-block ${className}`}>{children}</code>;
      }
      return <code className="md-code">{children}</code>;
    },
    pre({ children }) {
      return <pre className="md-pre">{children}</pre>;
    },
    table({ children }) {
      return <table className="md-table">{children}</table>;
    },
    thead({ children }) {
      return <thead>{children}</thead>;
    },
    tbody({ children }) {
      return <tbody>{children}</tbody>;
    },
    tr({ children }) {
      return <tr>{children}</tr>;
    },
    th({ children }) {
      return <th className="md-th">{walk(children)}</th>;
    },
    td({ children }) {
      return <td className="md-td">{walk(children)}</td>;
    },
    hr() {
      return <hr className="md-hr" />;
    },
  };

  return (
    <div className="answer-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        skipHtml
        disallowedElements={["script", "iframe", "style"]}
        urlTransform={chatUrlTransform}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function buildCiteNumbers(text: string, sourceIds: string[]): Map<string, number> {
  const numbers = new Map<string, number>();
  let next = 1;
  CITE_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CITE_RE.exec(text)) !== null) {
    const id = match[1].toLowerCase();
    if (!numbers.has(id)) numbers.set(id, next++);
  }
  for (const sid of sourceIds) {
    const norm = sid.toLowerCase();
    if (!numbers.has(norm)) numbers.set(norm, next++);
  }
  return numbers;
}
