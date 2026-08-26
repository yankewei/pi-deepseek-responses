import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

const RESPONSES_API = "openai-responses";

// DeepSeek's Responses API currently documents this model ID. Provider
// prefixes such as "deepseek/" and "deepseek-ai/" are ignored for matching,
// while the original ID is preserved in the outgoing request.
const DEEPSEEK_RESPONSES_MODEL_IDS = new Set([
	"deepseek-v4-flash",
]);

// Server-side web search tool name accepted by the DeepSeek Responses API.
const WEB_SEARCH_TOOL_TYPE = "web_search";

// Cap the standalone search request so the search summary stays compact.
const WEB_SEARCH_MAX_OUTPUT_TOKENS = 4096;

// The runtime preserves this field, although the public ProviderModelConfig
// type in the current pi release does not expose it yet.
type ProviderModelConfigWithSamplingParams = ProviderModelConfig & {
	samplingParams?: Record<string, unknown>;
};

function modelLeafId(modelId: string): string {
	const slash = modelId.lastIndexOf("/");
	return modelId.slice(slash + 1).toLowerCase();
}

export function isDeepSeekResponsesModel(model: Pick<Model<Api>, "id">): boolean {
	return DEEPSEEK_RESPONSES_MODEL_IDS.has(modelLeafId(model.id));
}

function toProviderModelConfig(model: Model<Api>): ProviderModelConfigWithSamplingParams {
	const useResponsesApi = isDeepSeekResponsesModel(model);

	return {
		id: model.id,
		name: model.name,
		api: useResponsesApi ? RESPONSES_API : model.api,
		baseUrl: model.baseUrl,
		reasoning: model.reasoning,
		thinkingLevelMap: model.thinkingLevelMap,
		input: model.input,
		cost: model.cost,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
		samplingParams: model.samplingParams,
		headers: model.headers,
		compat: useResponsesApi
			? {
					// DeepSeek accepts the developer role in Responses requests.
					supportsDeveloperRole: true,
					// DeepSeek manages prompt caching itself and ignores this field.
					supportsLongCacheRetention: false,
					// Avoid emitting strict: true unless a provider-specific override opts in.
					supportsStrictMode: false,
				}
			: model.compat,
	};
}

function groupModelsByProvider(models: readonly Model<Api>[]): Map<string, Model<Api>[]> {
	const grouped = new Map<string, Model<Api>[]>();

	for (const model of models) {
		const providerModels = grouped.get(model.provider) ?? [];
		providerModels.push(model);
		grouped.set(model.provider, providerModels);
	}

	return grouped;
}

// ---------------------------------------------------------------------------
// Client-side web_search tool.
//
// The model decides when to call this tool. It proxies the call to the
// provider's native server-side web search (the DeepSeek Responses API
// executes the search within the same request and answers from the results),
// then hands the findings back to the model as a tool result.
// ---------------------------------------------------------------------------

const WEB_SEARCH_PARAMS = Type.Object({
	query: Type.String({
		description: "The question or topic to search the web for.",
	}),
});

type WebSearchParams = Static<typeof WEB_SEARCH_PARAMS>;

interface ProviderWebSearchResult {
	text: string;
	details: {
		query: string;
		model: string;
		provider: string;
		searchCallIds: string[];
		usage: unknown;
	};
}

async function runProviderWebSearch(
	model: Model<Api>,
	query: string,
	ctx: ExtensionContext,
	signal: AbortSignal | undefined,
): Promise<ProviderWebSearchResult> {
	const auth = await ctx.modelRegistry.getProviderAuth(model.provider);
	if (!auth?.auth) {
		throw new Error(`No resolved credentials available for provider "${model.provider}".`);
	}

	const baseUrl = auth.auth.baseUrl ?? model.baseUrl;
	const endpoint = new URL("responses", baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();

	const headers: Record<string, string> = {
		"Content-Type": "application/json",
		...(auth.auth.headers ?? {}),
	};
	if (auth.auth.apiKey && !Object.keys(headers).some((key) => key.toLowerCase() === "authorization")) {
		headers["Authorization"] = `Bearer ${auth.auth.apiKey}`;
	}

	const payload = {
		model: model.id,
		instructions:
			"Use the web_search tool to search for the user's query. " +
			"Answer directly and concisely from the search results, citing the retrieved sources.",
		input: [{ role: "user", content: [{ type: "input_text", text: query }] }],
		tools: [{ type: WEB_SEARCH_TOOL_TYPE }],
		tool_choice: "required",
		max_output_tokens: WEB_SEARCH_MAX_OUTPUT_TOKENS,
		stream: false,
	};

	let response: Response;
	try {
		response = await fetch(endpoint, {
			method: "POST",
			headers,
			body: JSON.stringify(payload),
			signal,
		});
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		throw new Error(`Web search request failed for provider "${model.provider}": ${reason}`);
	}

	const data = (await response.json().catch(() => ({}))) as {
		status?: string;
		error?: { message?: string; type?: string; code?: string };
		output?: Array<{
			type: string;
			content?: Array<{ type: string; text?: string }>;
		}>;
		usage?: unknown;
	};

	if (!response.ok) {
		const message = data.error?.message ?? `HTTP ${response.status} ${response.statusText}`;
		throw new Error(
			`Provider "${model.provider}" does not support native web search or rejected the search request: ${message}`,
		);
	}
	if (data.status && data.status !== "completed") {
		throw new Error(`Web search did not complete for provider "${model.provider}" (status: ${data.status}).`);
	}

	const textParts: string[] = [];
	const searchCallIds: string[] = [];
	for (const item of data.output ?? []) {
		if (item.type === "message") {
			for (const part of item.content ?? []) {
				if (typeof part.text === "string" && part.text.length > 0) {
					textParts.push(part.text);
				}
			}
		} else if (item.type === "web_search_call" || item.type === "web_search") {
			searchCallIds.push(String((item as { id?: unknown }).id ?? ""));
		}
	}

	const text = textParts.join("\n\n").trim();
	if (!text) {
		throw new Error(
			`Provider "${model.provider}" executed the search but returned no readable results ` +
				`(search calls: ${searchCallIds.length || "none"}).`,
		);
	}

	return {
		text,
		details: {
			query,
			model: model.id,
			provider: model.provider,
			searchCallIds,
			usage: data.usage,
		},
	};
}

function supportedModelDescription(): string {
	return [...DEEPSEEK_RESPONSES_MODEL_IDS].sort().join(", ");
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
			const model = ctx.model;
			if (!model || model.api !== RESPONSES_API || !isDeepSeekResponsesModel(model)) {
				throw new Error(
					`Current model (${model ? `${model.provider}/${model.id}` : "unknown"}) does not support ` +
						`provider-side web search. Supported models: ${supportedModelDescription()}.`,
				);
			}

			const result = await runProviderWebSearch(model, params.query, ctx, signal);
			return {
				content: [{ type: "text", text: result.text }],
				details: result.details,
			};
		},
	});
}