# pi DeepSeek Responses

An extension package for [pi](https://github.com/earendil-works/pi) that enables DeepSeek's Responses API for compatible DeepSeek models without changing the selected provider.

When the extension starts, it:

1. Finds supported DeepSeek model IDs across all configured providers.
2. Re-registers each affected provider with the same complete model list.
3. Changes only the matching DeepSeek models to `openai-responses`.
4. Adds the server-side `{ "type": "web_search" }` tool to those Responses requests.

The original provider ID is retained. Therefore its existing `baseUrl`, API key or OAuth credentials, headers, and billing path are retained as well. The provider must expose a Responses-compatible `POST /responses` endpoint and forward DeepSeek's Responses format.

## Install from npm

After publishing, install it globally with:

```bash
pi install npm:@yankewei/pi-deepseek-responses
```

To install a specific version:

```bash
pi install npm:@yankewei/pi-deepseek-responses@0.1.1
```

Pi reads the `pi.extensions` manifest from the package and loads `src/index.ts` automatically.

## Local development

```bash
npm install
npm run check
```

Load the extension with:

```bash
pi -e /Users/yankewei/Documents/github/pi-deepseek-responses/src/index.ts
```

The package can also be loaded directly during development:

```bash
pi -e /Users/yankewei/Documents/github/pi-deepseek-responses
```

Before releasing, verify the package contents:

```bash
npm run check
npm run pack:check
```

## Release from Git

The GitHub repository is the source of published versions. A release is created by
updating the package version and pushing the generated tag:

```bash
npm version patch
git push origin main --follow-tags
```

The `vX.Y.Z` tag starts `.github/workflows/release.yml`, which runs the checks and
publishes the matching version to npm through GitHub Actions Trusted Publishing.
The scoped npm package must have a trusted publisher configured for:

- Repository: `yankewei/pi-deepseek-responses`
- Workflow: `release.yml`
- Permission: `npm publish`

For a new npm package, the configuration can also be created with:

```bash
npm trust github @yankewei/pi-deepseek-responses \
  --repository yankewei/pi-deepseek-responses \
  --file release.yml \
  --allow-publish \
  --yes
```

## Supported models

The extension currently matches this model ID, including provider-prefixed variants such as `deepseek/deepseek-v4-flash`:

- `deepseek-v4-flash`

The original model ID is sent unchanged. This matters for providers such as OpenRouter that require a provider-qualified model ID.

## Web search behavior

The extension makes `web_search` available to the model and leaves `tool_choice` as `auto`. The model decides whether to search. It does not force a search on every request.

If a provider does not implement or forward the Responses API and DeepSeek's `web_search` tool, the request will fail or the tool may be ignored; the extension cannot make a Chat Completions-only endpoint behave as a Responses endpoint.

## Version

The package is pinned to the pi `0.84.3` API packages used by the source workspace. Update the two `@earendil-works` dependencies together when upgrading pi.
