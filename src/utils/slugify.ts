import kebabcase from "lodash.kebabcase";
import slugify from "slugify";

const hasNonLatin = (str: string): boolean => /[^\x00-\x7F]/.test(str);

/**
 * Slugify a string using a hybrid approach:
 * - Latin strings: slugify (e.g. "E2E Testing" → "e2e-testing")
 * - Strings with non-Latin chars: lodash.kebabcase (preserves non-Latin chars)
 */
export const slugifyStr = (str: string): string => {
  if (hasNonLatin(str)) {
    return kebabcase(str);
  }
  return slugify(str, { lower: true });
};

export const slugifyAll = (arr: string[]) => arr.map(str => slugifyStr(str));

/**
 * Aggressive slug for filesystem/image names (no punctuation at all).
 * e.g. "Colab vs Laptop: A Guide?" → "colab-vs-laptop-a-guide"
 */
export const imageSlug = (str: string): string => {
  if (hasNonLatin(str)) {
    return kebabcase(str).replace(/[^a-zA-Z0-9]+/g, "-");
  }
  return slugify(str, { lower: true, strict: true });
};
