import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

function closeUnbalancedFence(text: string): string {
  const fenceCount = (text.match(/```/g) || []).length;
  return fenceCount % 2 === 1 ? text + "\n```" : text;
}

const components: Components = {
  p({ children }) { return <p className="md-p">{children}</p>; },
  h1({ children }) { return <h2 className="md-h2">{children}</h2>; },
  h2({ children }) { return <h2 className="md-h2">{children}</h2>; },
  h3({ children }) { return <h3 className="md-h3">{children}</h3>; },
  h4({ children }) { return <h4 className="md-h4">{children}</h4>; },
  h5({ children }) { return <h4 className="md-h4">{children}</h4>; },
  h6({ children }) { return <h4 className="md-h4">{children}</h4>; },
  ul({ children }) { return <ul className="md-list">{children}</ul>; },
  ol({ children }) { return <ol className="md-list md-list-ordered">{children}</ol>; },
  li({ children }) { return <li className="md-li">{children}</li>; },
  blockquote({ children }) { return <blockquote className="md-quote">{children}</blockquote>; },
  strong({ children }) { return <strong className="md-bold">{children}</strong>; },
  em({ children }) { return <em className="md-italic">{children}</em>; },
  a({ href, children }) {
    return (
      <a className="md-url" href={href ?? "#"} target="_blank" rel="noopener noreferrer" title={href}>
        {children}
      </a>
    );
  },
  code({ className, children }) {
    const isBlock = !!(className && className.startsWith("language-"));
    if (isBlock) return <code className={`md-code-block ${className}`}>{children}</code>;
    return <code className="md-code">{children}</code>;
  },
  pre({ children }) { return <pre className="md-pre">{children}</pre>; },
  table({ children }) { return <table className="md-table">{children}</table>; },
  thead({ children }) { return <thead>{children}</thead>; },
  tbody({ children }) { return <tbody>{children}</tbody>; },
  tr({ children }) { return <tr>{children}</tr>; },
  th({ children }) { return <th className="md-th">{children}</th>; },
  td({ children }) { return <td className="md-td">{children}</td>; },
  hr() { return <hr className="md-hr" />; },
};

export function MarkdownSnippet({ text }: { text: string }) {
  const safe = closeUnbalancedFence(text);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      skipHtml
      disallowedElements={["script", "iframe", "style"]}
      components={components}
    >
      {safe}
    </ReactMarkdown>
  );
}
