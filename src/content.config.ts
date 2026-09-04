import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import { glob } from "astro/loaders";
import config from "@/config";

export const BLOG_PATH = "src/content/posts";

const posts = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      author: z.string().default(config.site.author),
      pubDatetime: z.date(),
      modDatetime: z.date().optional().nullable(),
      title: z.string(),
      featured: z.boolean().optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).default(["others"]),
      ogImage: image().or(z.string()).optional(),
      description: z.string(),
      canonicalURL: z.string().optional(),
      hideEditPost: z.boolean().optional(),
      timezone: z.string().optional(),
    }),
});

const pages = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/pages" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    ogImage: z.string().optional(),
    canonicalURL: z.string().optional(),
  }),
});

const paper = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/paper" }),
  schema: z.object({
    title: z.string(),
    venue: z.string(),
    year: z.coerce.number(),
    authors: z.array(z.string()).optional(),
    doi: z.string().optional(),
    url: z.string().optional(),
    abstract: z.string().optional(),
    citation: z.string().optional(),
  }),
});

const book = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/book" }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      pubDatetime: z.date(),
      modDatetime: z.date().optional().nullable(),
      description: z.string(),
      draft: z.boolean().optional(),
      chapter: z.number().int().positive(),
      bookId: z.string(),
      tags: z.array(z.string()).default([]),
      ogImage: image().or(z.string()).optional(),
      timezone: z.string().optional(),
    }),
});

export const collections = { posts, pages, paper, book };
