# kanadakurniawan.com — Blog & Research

Personal blog and research site of **Kanada Kurniawan** — AI × Meteorology
(researcher at BMKG, Stasiun Meteorologi Kelas III Maritim Dwikora, Pontianak).

Built with [Astro](https://astro.build), deployed for free to **GitHub Pages**.

## Features

- 📝 Blog posts (Markdown/MDX) on deep learning, meteorology, and research notes
- 📖 Publications page (paper list from Google Scholar)
- 🧮 KaTeX math rendering (`$...$` inline, multi-line `$$ ... $$` display)
- 🗞️ RSS feed (`/rss.xml`) + sitemap (`/sitemap-index.xml`)
- 📱 Responsive, dark-mode aware
- 🔄 GitHub Actions: auto-build & deploy on push to `main`

## Tech Stack

- **Astro 5** + MDX + TypeScript
- remark-math + rehype-katex for equations
- @astrojs/rss + @astrojs/sitemap
- Static deployment via GitHub Actions

## Local Development

```sh
npm install     # first time only
npm run dev     # preview at http://localhost:4321
npm run build   # production build to ./dist/
npm run preview # preview the production build
```

## Project Structure

```
site/
├── src/
│   ├── content/
│   │   ├── posts/          # blog posts (Markdown/MDX)
│   │   └── publications/   # publication records (Markdown)
│   ├── layouts/            # base layout
│   ├── components/         # header, footer, post card, avatar
│   ├── pages/              # home, blog, about, publications, rss, sitemap
│   ├── scripts/            # build/verification helpers
│   ├── config.ts           # site metadata & nav
│   └── styles/             # global CSS
├── public/                 # static assets (favicon, images)
└── .github/workflows/      # GitHub Pages deploy
```

## Adding Content

1. Create a Markdown/MDX file in `src/content/posts/` with frontmatter:

   ```md
   ---
   title: "Post Title"
   description: "One-line summary"
   pubDate: 2026-09-01
   categories: ["Deep Learning"]
   tags: ["neural-network"]
   draft: false
   ---
   ```

2. The filename becomes the URL: `my-post.md` → `/blog/my-post/`.
3. Push to `main` — GitHub Actions builds and deploys automatically.

Some posts are synced from the *Pengantar Deep Learning untuk Meteorologi* book repository
via its build script before publication, then built and pushed here.

## License

Content © Kanada Kurniawan. Source code under the MIT License unless noted.