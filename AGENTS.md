# Awuawu Blog Agent Instructions

## 1. Project

This repository is the source code of the awuawu personal blog.

Technology stack:

- Hugo
- PaperMod
- GitHub Pages
- HTML / CSS / JavaScript where necessary

The existing Hugo architecture must be preserved.

Do NOT migrate the project to:

- React
- Next.js
- Vue
- Nuxt
- Astro
- another static-site framework

unless explicitly requested by the user.

---

## 2. Agent Role

You are responsible for implementation and technical verification.

The user is primarily the final reviewer.

Do not ask the user to perform routine development steps that you can execute yourself.

For normal development tasks:

1. inspect
2. plan
3. implement
4. build
5. launch locally
6. visually inspect
7. test
8. fix discovered issues
9. repeat verification
10. present final results

Only ask the user when a decision genuinely requires subjective preference or authorization.

---

## 3. Source-of-truth directories

Prioritize:

- hugo.toml
- content/
- layouts/
- assets/
- static/
- .github/workflows/

Do NOT treat public/ as source code.

`public/` is generated output.

---

## 4. PaperMod rule

Do NOT directly modify:

themes/PaperMod/

`themes/PaperMod` is a Git submodule. By default, do not directly modify any
files in it. If a future task genuinely requires modifying the submodule, first
explain the reason and obtain explicit user authorization.

Prefer Hugo override mechanisms:

- layouts/
- assets/css/extended/
- content/
- static/

This ensures future PaperMod upgrades do not destroy customizations.

Do not attempt to resolve the current `git submodule status` Win32 error 5 as
part of blog optimization. It is a separate environment issue and does not
block work: `themes/PaperMod` has already been confirmed to exist and be
readable.

---

## 5. Content safety

Do not invent:

- personal information
- education history
- publications
- projects
- employment
- awards
- research results

unless those facts already exist in repository content or are explicitly supplied by the user.

Use placeholders when information is unknown.

---

## 6. GitHub Pages

This site is deployed through GitHub Pages.

Current project configuration:

- Configuration file: `hugo.toml`
- Workflow: `.github/workflows/hugo.yaml`
- GitHub Pages repository subpath: `/awuawu-blog/`
- Current `baseURL`: `https://awuawu315.github.io/awuawu-blog/`

`.github/workflows/hugo.yaml` currently obtains the Pages base URL through
`actions/configure-pages` and passes it to the Hugo build using a command like:

```text
--baseURL "${{ steps.pages.outputs.base_url }}/"
```

Before modifying routing, URLs, assets, or navigation:

1. inspect `hugo.toml`
2. inspect `.github/workflows/hugo.yaml`
3. identify the configured `baseURL`
4. preserve deployment under the `/awuawu-blog/` repository subpath

When modifying routes, navigation, CSS/JS resources, images, permalinks,
canonical URLs, search indexes, or custom layouts, do not assume the site is
deployed at the domain root `/`. Maintain GitHub Pages repository-subpath
compatibility and do not introduce absolute-root URL assumptions that break
deployment.

---

## 7. Build verification

Before declaring a task complete, run the appropriate Hugo production build.

The production build must complete without errors.

The current local Hugo version is `0.152.2 extended`, and GitHub Actions uses
`0.152.2`; local and CI versions currently match. Do not independently upgrade
or downgrade Hugo without a clear technical need and explicit authorization.

## 7.1 Git operations

During normal development, read-only Git commands such as `git status`,
`git diff`, and `git log` are allowed.

Before modifying files, inspect existing uncommitted changes and avoid
overwriting user work.

Unless the user explicitly authorizes it, do not:

- run `git commit`
- run `git push`
- run `git reset --hard`
- run `git clean`
- force-overwrite existing user changes

---

## 8. Browser verification

For every UI change, use Playwright interactive whenever available.

Launch the local Hugo development server and inspect the rendered site.

At minimum verify:

- Home
- Blog / Posts
- individual article
- About
- Archives
- Search

if those routes exist.

---

## 9. Responsive verification

Verify at least:

Desktop:
- 1440 × 900

Tablet:
- 768 × 1024

Mobile:
- 390 × 844

Check:

- navigation
- typography
- horizontal overflow
- margins
- card layout
- article width
- code blocks
- TOC
- footer
- dark mode
- light mode

---

## 10. Functional acceptance criteria

Before reporting completion:

- Hugo production build succeeds
- navigation contains no known broken links
- no unexpected 404 pages
- no horizontal scrolling on mobile
- light mode is readable
- dark mode is readable
- article pages remain usable
- search works if enabled
- archives work if enabled
- existing posts are preserved
- GitHub Pages path compatibility is preserved

---

## 11. Visual quality

The target visual direction is:

- professional
- modern
- restrained
- developer/researcher oriented
- strong typography
- generous but controlled whitespace
- subtle borders
- minimal visual noise
- responsive

Avoid excessive:

- gradients
- neon effects
- particles
- animations
- glassmorphism
- decorative elements

unless explicitly requested.

---

## 12. Completion report

Do not simply say "done."

At completion provide:

1. files changed
2. important implementation decisions
3. build command and result
4. browser routes tested
5. responsive sizes tested
6. remaining known issues
7. screenshots or visual evidence when possible
8. git diff summary

Clearly distinguish:

- verified
- inferred
- not tested

Every verification item must use one of these fixed status categories:

- 已验证 (Verified)
- 未验证 (Not Tested / Unverified)
- 推断 (Inferred)

Never present an inferred result as verified.
