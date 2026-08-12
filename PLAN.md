# Blog Optimization V1

## Goal

Upgrade the existing Hugo + PaperMod blog into a polished personal
technical/research blog while preserving the current architecture.

## Milestone 1

Audit existing site:

- `hugo.toml`
- `.github/workflows/hugo.yaml`
- navigation
- PaperMod configuration
- `content/`
- `layouts/`
- `assets/`
- `static/`
- GitHub Pages deployment
- GitHub Pages repository-subpath compatibility
- broken links

Produce findings before major edits.

## Milestone 2

Improve site structure:

- Home
- About
- Archives
- Search

Fix existing 404 routes.

## Milestone 3

Improve homepage:

- hero
- personal introduction placeholder
- latest posts
- topic navigation
- project/research entry placeholders where appropriate

Do not invent personal content.

Do not generate real personal experience, publications, projects, awards, or
other personal facts. Use placeholders, neutral copy, or do not display the
section when content is unknown.

## Milestone 4

Improve the `/posts/` blog list page:

- page title and introduction
- article list hierarchy
- titles, excerpts, dates, and reading time
- tags/categories when data exists
- card or list spacing
- hover states
- desktop and mobile reading experience

Do not remove existing articles for visual effect.

## Milestone 5

Repair or create the About route. The current project has no effective About
content page. Use neutral placeholder content when real personal information is
insufficient; do not invent facts. Ensure the About navigation no longer leads
to a 404.

## Milestone 6

Create or repair the Archives page. Verify that the route is accessible,
archives are displayed, and GitHub Pages repository-subpath handling is
correct.

## Milestone 7

Create or repair the Search page. When using PaperMod search, check the
required output configuration and Fuse.js/index JSON requirements, and ensure
search works under the GitHub Pages repository subpath.

## Milestone 8

Improve article reading experience:

- typography
- table of contents
- breadcrumbs
- reading time
- code copy
- previous/next navigation
- code block styling

Do not cause PaperMod article features to regress when overriding layouts.

## Milestone 9

Light and Dark Mode must both receive actual browser inspection. Check:

- text contrast
- backgrounds
- borders
- cards
- links
- code blocks
- navigation
- footer

Do not validate only one theme.

## Milestone 10

Responsive design:

Test at least:

- Desktop: 1440 × 900
- Tablet: 768 × 1024
- Mobile: 390 × 844

Check horizontal overflow, navigation, typography, container width,
margins/padding, articles, code blocks, TOC, footer, and cards.

## Milestone 11

Visual QA:

Use Playwright to inspect the actual rendered website.

Iterate until there are no obvious layout issues.

Browser verification must include:

- `/`
- `/posts/`
- one real article page
- `/about/`
- `/archives/`
- `/search/`

If a page is not implemented in the final result, explicitly report it as
unverified/not implemented; never present it as passing.

## Milestone 12

Production validation:

Run a Hugo production build using the current Hugo `0.152.2 extended` where
possible.

Verify that the build has no errors or new critical warnings, existing articles
remain present, main navigation has no known 404s, and GitHub Pages
repository-subpath handling remains compatible.

## Done when

- production build succeeds
- Home passes actual browser verification
- Posts list page passes actual browser verification
- at least one article page passes verification
- About has no known 404
- Archives work
- Search works if enabled
- Desktop 1440 × 900 is verified
- Tablet 768 × 1024 is verified
- Mobile 390 × 844 is verified
- Light Mode is verified
- Dark Mode is verified
- no known mobile horizontal overflow
- current content is preserved
- GitHub Pages `/awuawu-blog/` subpath compatibility is preserved
- final report distinguishes: 已验证 / 未验证 / 推断
