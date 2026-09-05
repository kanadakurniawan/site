import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { renderOgWebp } from "@/utils/og-template";
import type { OgVariant } from "@/utils/og-template";
import { imageSlug } from "@/utils/slugify";
import config from "@/config";

type Props = {
  title: string;
  author: string;
  meta?: string;
  variant?: OgVariant;
};

export async function getStaticPaths() {
  if (!config.features.dynamicOgImage) {
    return [];
  }

  const posts = await getCollection("posts").then(p =>
    p.filter(({ data }) => !data.draft && !data.ogImage)
  );

  return posts.map(post => ({
    // Same descriptive filename as the .jpg variant.
    params: { name: imageSlug(post.data.title) },
    props: {
      title: post.data.title,
      author: post.data.author,
      meta: post.data.tags?.[0],
      variant: post.data.ogVariant,
    } satisfies Props,
  }));
}

export const GET: APIRoute = async ({ props, url }) => {
  if (!config.features.dynamicOgImage) {
    return new Response(null, { status: 404, statusText: "Not found" });
  }

  const { title, author, meta, variant } = props as Props;

  const { buffer, contentType } = await renderOgWebp(
    {
      title,
      meta,
      footerLeft: `by ${author}`,
      footerRight: new URL(config.site.url).hostname,
      variant,
    },
    url
  );

  return new Response(new Uint8Array(buffer), {
    headers: {
      "Content-Type": contentType,
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
};