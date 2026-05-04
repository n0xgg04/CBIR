"use client";

import {
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
  useCallback,
  useId,
  useRef,
  useState,
} from "react";

export interface DropzoneProps {
  /** Accept attribute forwarded to `<input type="file">` (e.g. `image/*`). */
  accept?: string;
  /** When true, multiple files can be selected at once. */
  multiple?: boolean;
  /** Disable interactions while a request is in flight. */
  disabled?: boolean;
  /** Fired with the list of files the user dropped or picked. */
  onFiles: (files: File[]) => void;
  /** Optional label rendered above the dropzone. */
  label?: ReactNode;
  /** Optional helper text rendered inside. */
  hint?: ReactNode;
}

export function Dropzone({
  accept = "image/*",
  multiple = false,
  disabled = false,
  onFiles,
  label,
  hint,
}: DropzoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list || list.length === 0) {
        return;
      }
      const files = Array.from(list);
      onFiles(multiple ? files : files.slice(0, 1));
    },
    [multiple, onFiles],
  );

  const onDrop = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setDragging(false);
      if (disabled) {
        return;
      }
      handleFiles(event.dataTransfer?.files ?? null);
    },
    [disabled, handleFiles],
  );

  const onDragOver = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (disabled) {
        return;
      }
      setDragging(true);
    },
    [disabled],
  );

  const onDragLeave = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setDragging(false);
    },
    [],
  );

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      handleFiles(event.target.files);
      // Allow re-selecting the same file later.
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    },
    [handleFiles],
  );

  const stateClass = disabled
    ? "opacity-60 cursor-not-allowed"
    : isDragging
      ? "border-brand bg-sky-50"
      : "hover:border-brand";

  return (
    <div className="flex flex-col gap-2">
      {label !== undefined ? (
        <span className="text-sm font-medium text-slate-700">{label}</span>
      ) : null}
      <label
        data-testid="dropzone"
        htmlFor={inputId}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDragEnd={onDragLeave}
        className={`flex min-h-[12rem] flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600 transition focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/30 ${stateClass}`}
      >
        <span className="text-base font-semibold text-slate-800">
          {isDragging ? "Thả ảnh vào đây" : "Kéo ảnh vào đây"}
        </span>
        <span className="text-sm text-slate-500">hoặc bấm để chọn tệp</span>
        {hint !== undefined ? (
          <span className="mt-1 text-xs text-slate-500">{hint}</span>
        ) : null}
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={accept}
          multiple={multiple}
          disabled={disabled}
          onChange={onChange}
          className="sr-only"
        />
      </label>
    </div>
  );
}
