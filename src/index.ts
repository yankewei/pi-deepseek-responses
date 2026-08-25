import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const RESPONSES_API = "openai-responses";

// DeepSeek's Responses API currently documents this model ID. Provider
// prefixes such as "deepseek/" and "deepseek-ai/" are ignored for matching,
// while the original ID is preserved in the outgoing request.
const DEEPSEEK_RESPONSES_MODEL_IDS = new Set([
	"deepseek-v4-flash",
]);

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

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function addWebSearchTool(payload: unknown): unknown {
	if (!isRecord(payload)) return payload;

	const tools = Array.isArray(payload.tools) ? payload.tools : [];
	const hasWebSearch = tools.some(
		(tool) =>
			isRecord(tool) &&
			(tool.type === "web_search" || tool.type === "web_search_2025_08_26"),
	);
	if (hasWebSearch) return payload;

	return {
		...payload,
		tools: [...tools, { type: "web_search" }],
	};
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

	pi.on("before_provider_request", (event, ctx) => {
		const model = ctx.model;
		if (!model || model.api !== RESPONSES_API || !isDeepSeekResponsesModel(model)) return;

		// Leave tool_choice as auto. DeepSeek can decide when web search is useful.
		return addWebSearchTool(event.payload);
	});
}
