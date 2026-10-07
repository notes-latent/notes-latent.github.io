# Notes on Retrieval & Recommendation

A small Markdown notebook for GitHub Pages. No analytics, comments, external fonts, or client-side scripts are included. Drafts stay out of the generated site, but are visible in a public source repository.

## Publish once

1. Create a separate pseudonymous GitHub account. Avoid reusing a recognizable handle, photo, bio, or links to your other accounts.
2. In **Settings → Emails**, enable **Keep my email addresses private** and **Block command line pushes that expose my email**. Copy the exact noreply address GitHub displays.
3. Create an empty public repository named `YOUR-USERNAME.github.io`, with default branch `main`.
4. Upload this project's source files, including `.github/workflows/pages.yml`. Do not upload `_site`, local caches, or credentials. Browser commits should use the anonymous account with email privacy already enabled.
5. In the repository, choose **Settings → Pages → Build and deployment → Source → GitHub Actions**. Run **Actions → Publish notebook → Run workflow** if the initial upload ran before Pages was enabled.
6. Once deployment succeeds, open `https://YOUR-USERNAME.github.io`.

## Write a note

Create `_posts/YYYY-MM-DD-short-title.md` in GitHub and commit:

```markdown
---
layout: post
title: "Your note title"
description: "Optional one-line summary"
---

Your Markdown goes here.
```

The homepage updates automatically after deployment. The file date must not be in the future (the site uses UTC). Code fences, tables, images, and ordinary Markdown work without additional configuration. For formulas, use inline code or fenced text initially; browser-rendered LaTeX is not configured.

The `_drafts/semantic-id.md` file is a writing template. To publish it, replace the prompts with your notes and move it to `_posts/YYYY-MM-DD-semantic-id.md`. Do not place private drafts in a public repository: excluding a draft from the website does not hide its source or history.

## Local commits, only if needed

Run these inside this repository **before the first commit**, replacing both placeholders:

```sh
git init -b main
git config --local user.name "YOUR-PSEUDONYM"
git config --local user.email "YOUR-EXACT-GITHUB-NOREPLY-ADDRESS"
git config --local user.useConfigOnly true
git config --local commit.gpgsign false
```

This prevents inheriting your usual author identity and disables inherited commit signing that could link to a known key. Use only the anonymous account to push; do not use work credentials. Do not share passwords or tokens in chat.

Public history can include author and committer names, email addresses, timestamps, old content, and signatures. Before pushing, check both author and committer metadata with `git log --format=fuller`. Deleting text in a later commit does not remove earlier versions. A noreply address hides your email, but still identifies the GitHub account: use the separate account throughout.

## Local preview (optional)

```sh
bundle install
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000`. To preview the writing template locally, add `--drafts`; never add that flag to the publishing workflow.

## Privacy boundaries

The site's templates contain no author identity, account links, tracking, or third-party assets. GitHub still hosts the repository and website and receives visitors' requests. This is pseudonymous publishing, not guaranteed anonymity. Review notes, screenshots, attachment metadata, external image URLs, and examples for names, employers, internal metrics, or distinctive project details before publishing.

Official guidance: [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), [commit email privacy](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).
