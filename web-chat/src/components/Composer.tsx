import { useEffect, useRef, useState } from "react";

export function Composer({
  onSubmit,
  disabled,
  onActivity,
}: {
  onSubmit: (text: string) => void;
  disabled: boolean;
  onActivity?: (active: boolean) => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = `${Math.min(ref.current.scrollHeight, 220)}px`;
    }
    onActivity?.(value.length > 0);
  }, [value, onActivity]);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    onActivity?.(false);
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={ref}
        placeholder="Preguntale a Synapse…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        rows={1}
        disabled={disabled}
        autoFocus
      />
      <button type="submit" disabled={disabled || value.trim().length === 0} aria-label="Enviar">
        {disabled ? "…" : "↑"}
      </button>
    </form>
  );
}
