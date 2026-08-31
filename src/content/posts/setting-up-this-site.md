---
title: "Setting Up This Site"
description: "How this Astro blog is structured, deployed for free on GitHub Pages, and how to add new posts."
pubDate: 2026-08-30
categories: ["Meta"]
tags: ["astro", "github pages", "static site"]
---

This site is built with [Astro](https://astro.build) and deployed for free to GitHub Pages. This
post documents the workflow so future-me (and contributors) know how to add content.

## Adding a post

Create a Markdown or MDX file in `src/content/posts/`:

```md
---
title: "My Post Title"
description: "One-line summary shown on the blog index."
pubDate: 2026-09-01
categories: ["AI"]
tags: ["tag-one", "tag-two"]
---

Content here. Inline math: $\frac{\partial u}{\partial t}$.

Display math:

$$
\frac{\partial u}{\partial t} = \nu \nabla^2 u
$$

```python
print("code blocks are syntax-highlighted")
```
```

The filename becomes the URL slug: `src/content/posts/my-post.md` publishes at
`/blog/my-post/`.

## Adding a publication

Create a file in `src/content/publications/`:

```md
---
title: "Title of the paper"
venue: "Some Conference or Journal"
year: 2026
doi: "10.1234/example"
url: "https://example.com/paper.pdf"
---

Optional notes or abstract go here.
```

## Math and code

LaTeX math works inline (`$...$`) and in display mode (`$$` on its own line around the
expression) via KaTeX. Code blocks are highlighted by Shiki, built into Astro.

## Build and deploy

- `npm run dev` — local preview.
- `npm run build` — build into `dist/`.
- Push to the `main` branch; the GitHub Actions workflow builds and deploys automatically.