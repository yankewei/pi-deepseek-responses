import { type Api, type Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	RESPONSES_API,
	isDeepSeekResponsesModel,
	isOpenAICodexWebSearchModel,
	supportedModelDescription,
	toProviderModelConfig,
} from "./models.ts";
import { WEB_SEARCH_PARAMS, runProviderWebSearch, type WebSearchParams } from "./web-search.ts";

export { isDeepSeekResponsesModel, isOpenAICodexWebSearchModel } from "./models.ts";

function groupModelsByProvider(models: readonly Model<Api>[]): Map<string, Model<Api>[]> {
	const grouped = new Map<string, Model<Api>[]>();

	for (const model of models) {
		const providerModels = grouped.get(model.provider) ?? [];
		providerModels.push(model);
		grouped.set(model.provider, providerModels);
	}

	return grouped;
}

export default function registerDeepSeekResponses(pi: ExtensionAPI): void {
	pi.on("session_start", (_event, ctx) => {
		const modelsByProvider = groupModelsByProvider(ctx.modelRegistry.getAll());

		for (const [provider, models] of modelsByProvider) {
			if (!models.some(isDeepSeekResponsesModel)) continue;

			// Register against the original provider ID. This preserves its base URL,
			// API key/OAuth resolution, headers, billing, and all non-DeepSeek models.
			// The complete model list is required because `models` replaces the
			// extension layer's model list for this provider.
			pi.registerProvider(provider, {
				models: models.map(toProviderModelConfig),
			});
		}
	});

	// The model decides when web search is useful and calls this tool explicitly.
	// The tool then uses the provider's native server-side web search (Responses
	// API) to look up the query and returns the findings as the tool result.
	pi.registerTool({
		name: "web_search",
		label: "Web Search",
		description:
			"Search the web through the current provider's native web search and return the retrieved findings. " +
			"Use it when the answer needs up-to-date, external, or factual information.",
		promptSnippet: "Search the web via the provider's native web search tool",
		promptGuidelines: [
			"Use web_search when the user asks about recent events, external facts, or information outside the model's knowledge.",
		],
		parameters: WEB_SEARCH_PARAMS,
		async execute(_toolCallId, params: WebSearchParams, signal, _onUpdate, ctx) {
			const query = params.query.trim();
			if (!query) throw new Error("Search query must not be empty.");

			const model = ctx.model;
			const supportsWebSearch =
				model !== undefined &&
				((model.api === RESPONSES_API && isDeepSeekResponsesModel(model)) ||
					isOpenAICodexWebSearchModel(model));
			if (!supportsWebSearch) {
				throw new Error(
					`Current model (${model ? `${model.provider}/${model.id}` : "unknown"}) does not support ` +
						`provider-side web search. Supported models: ${supportedModelDescription()}.`,
				);
			}

			const result = await runProviderWebSearch(model, query, ctx, signal);
			return {
				content: [{ type: "text", text: result.text }],
				details: result.details,
				usage: result.usage,
			};
		},
	});
}
