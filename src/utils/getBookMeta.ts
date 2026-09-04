import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const CONTENT_ROOT = new URL("../../content/book/", import.meta.url).pathname;

export type BookMeta = {
  title: string;
  tagline?: string;
  author?: string;
  totalChapters?: number;
  version?: string;
  repository?: string;
  doi?: string | null;
  isbn?: string | null;
};

export function getBookMeta(bookId: string): BookMeta | null {
  const candidates = [
    join(CONTENT_ROOT, bookId, "_book.json"),
    join(process.cwd(), "src", "content", "book", bookId, "_book.json"),
  ];
  for (const metaPath of candidates) {
    if (existsSync(metaPath)) {
      try {
        return JSON.parse(readFileSync(metaPath, "utf8")) as BookMeta;
      } catch {
        return null;
      }
    }
  }
  return null;
}
