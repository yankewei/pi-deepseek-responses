import { type Api, type Model } from "@earendil-works/pi-ai";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";

export const RESPONSES_API = "openai-responses";

// DeepSeek's Responses API currently documents this model ID. Provider
// prefixes such as "deepseek/" and "deepseek-ai/" are ignored for matching,
// while the original ID is preserved in the outgoing request.
const DEEPSEEK_RESPONSES_MODEL_IDS = new Set([
	"deepseek-v4-flash",
]);

const CODEX_RESPONSES_API = "openai-codex-responses";

// These models share Pi's Codex Responses transport. Keep the list explicit
// until Pi exposes a stable capability flag for hosted web search.
const OPENAI_CODEX_WEB_SEARCH_MODEL_IDS = new Set([
	"gpt-5.6-luna",
	"gpt-5.6-sol",
	"gpt-5.6-terra",
]);

// The Responses API has different effort semantics from DeepSeek's legacy
// Chat Completions model metadata. Keep this mapping local to the API adapter.
const DEEPSEEK_RESPONSES_THINKING_LEVEL_MAP: NonNullable<ProviderModelConfig["thinkingLevelMap"]> = {
	off: "none",
	minimal: "low",
	low: "low",
	medium: "high",
	high: "high",
	xhigh: "high",
	max: "max",
};

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

export function isOpenAICodexWebSearchModel(
	model: Pick<Model<Api>, "id" | "api" | "provider">,
): boolean {
	return (
		model.api === CODEX_RESPONSES_API &&
		model.provider === "openai-codex" &&
		OPENAI_CODEX_WEB_SEARCH_MODEL_IDS.has(modelLeafId(model.id))
	);
}

export function toProviderModelConfig(model: Model<Api>): ProviderModelConfigWithSamplingParams {
	const useResponsesApi = isDeepSeekResponsesModel(model);

	return {
		id: model.id,
		name: model.name,
		api: useResponsesApi ? RESPONSES_API : model.api,
		baseUrl: model.baseUrl,
		reasoning: model.reasoning,
		thinkingLevelMap: useResponsesApi ? DEEPSEEK_RESPONSES_THINKING_LEVEL_MAP : model.thinkingLevelMap,
		// The currently supported Responses model is text-only. Do not inherit
		// image capability from a provider catalog with broader metadata.
		input: useResponsesApi ? ["text"] : model.input,
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

export function supportedModelDescription(): string {
	return [
		...[...DEEPSEEK_RESPONSES_MODEL_IDS].map((id) => `${id} (Responses API)`),
		...[...OPENAI_CODEX_WEB_SEARCH_MODEL_IDS].map((id) => `openai-codex/${id}`),
	].sort().join(", ");
}
