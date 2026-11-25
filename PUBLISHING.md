# GitHub Pages

Site address: <https://ldomenichelli.github.io/linguistic_profiling_of_transformer/>

## First deployment

1. Open **Settings → Pages** in the repository.
2. Under **Build and deployment → Source**, select **GitHub Actions**.
3. Open **Actions → Deploy project site → Run workflow**.
4. Select `main`, enable `publish_publicly`, and run the workflow.
5. Wait for both the build and deployment to succeed.

The workflow is already included; no additional template is needed. It builds the static website in `docs/` without downloading models or the large LFS visualizations.

## Updates

Pushes to `main` run the checks and publish updates when Pages is enabled and the repository is public. Pull requests build and validate the website without deploying it. Manual runs deploy only when `publish_publicly` is enabled.

Before pushing changes:

```bash
make site
make check
git push origin main
```

If Pages has not been configured, the workflow uploads the checked site artifact and shows activation instructions in its summary.
