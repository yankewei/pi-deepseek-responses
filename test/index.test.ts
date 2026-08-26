import assert from "node:assert/strict";
import { test } from "node:test";
import { zstdDecompressSync } from "node:zlib";
import registerDeepSeekResponses, { isDeepSeekResponsesModel } from "../src/index.ts";
import { isOpenAICodexWebSearchModel } from "../src/index.ts";
import type { Api, Model } from "@earendil-works/pi-ai";
import { openAICodexResponsesApi } from "@earendil-works/pi-ai/api/openai-codex-responses.lazy";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Handler = (...args: any[]) => unknown;
type RegisteredTool = {
	name: string;
	execute: (...args: any[]) => Promise<any>;
};

function makeModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
	return {
		id: "deepseek-v4-flash",
		name: "DeepSeek V4 Flash",
		api: "openai-completions",
		provider: "deepseek",
		baseUrl: "https://api.deepseek.com",
		reasoning: true,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 384_000,
		...overrides,
	};
}

function makeCodexModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
	return {
		id: "gpt-5.6-luna",
		name: "GPT-5.6 Luna",
		api: "openai-codex-responses",
		provider: "openai-codex",
		baseUrl: "https://chatgpt.com/backend-api",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0.2, output: 1.2, cacheRead: 0.02, cacheWrite: 0.25 },
		contextWindow: 272_000,
		maxTokens: 128_000,
		thinkingLevelMap: { minimal: "low", xhigh: "xhigh", max: "max" },
		compat: {
			supportsOpenAIGrammarTools: true,
			supportsAdditionalTools: true,
			supportsToolSearch: true,
		},
		...overrides,
	};
}

async function decodeRequestBody(body: BodyInit | null | undefined, compressed: boolean): Promise<any> {
	if (typeof body === "string") return JSON.parse(body);
	assert.ok(body instanceof Uint8Array);
	const bytes = compressed ? zstdDecompressSync(body) : body;
	return JSON.parse(new TextDecoder().decode(bytes));
}

function setupExtension(): {
	handlers: Map<string, Handler>;
	providers: Array<{ name: string; config: any }>;
	tools: RegisteredTool[];
} {
	const handlers = new Map<string, Handler>();
	const providers: Array<{ name: string; config: any }> = [];
	const tools: RegisteredTool[] = [];
	const pi = {
		on(name: string, handler: Handler) {
			handlers.set(name, handler);
		},
		registerProvider(name: string, config: any) {
			providers.push({ name, config });
		},
		registerTool(tool: RegisteredTool) {
			tools.push(tool);
		},
	};

	registerDeepSeekResponses(pi as unknown as ExtensionAPI);
	return { handlers, providers, tools };
}

test("matches exact model IDs and provider-prefixed variants", () => {
	assert.equal(isDeepSeekResponsesModel({ id: "deepseek-v4-flash" }), true);
	assert.equal(isDeepSeekResponsesModel({ id: "openrouter/deepseek-v4-flash" }), true);
	assert.equal(isDeepSeekResponsesModel({ id: "deepseek-v4-pro" }), false);
	assert.equal(isOpenAICodexWebSearchModel(makeCodexModel()), true);
	assert.equal(isOpenAICodexWebSearchModel(makeCodexModel({ id: "gpt-5.6-sol" })), true);
	assert.equal(isOpenAICodexWebSearchModel(makeCodexModel({ id: "gpt-5.5" })), false);
	assert.equal(isOpenAICodexWebSearchModel(makeCodexModel({ api: "openai-responses" })), false);
});

test("registers the complete provider model list with Responses metadata", async () => {
	const { handlers, providers } = setupExtension();
	const target = makeModel({
		provider: "proxy",
		input: ["text", "image"],
		thinkingLevelMap: { minimal: null, low: "low", medium: null, high: "high", max: "max" },
	});
	const other = makeModel({ id: "other-model", name: "Other", provider: "proxy", reasoning: false });

	await handlers.get("session_start")?.({}, { modelRegistry: { getAll: () => [target, other] } });

	assert.equal(providers.length, 1);
	assert.equal(providers[0].name, "proxy");
	assert.equal(providers[0].config.models.length, 2);
	assert.equal(providers[0].config.models[0].api, "openai-responses");
	assert.deepEqual(providers[0].config.models[0].input, ["text"]);
	assert.deepEqual(providers[0].config.models[0].thinkingLevelMap, {
		off: "none",
		minimal: "low",
		low: "low",
		medium: "high",
		high: "high",
		xhigh: "high",
		max: "max",
	});
	assert.equal(providers[0].config.models[1].api, "openai-completions");
});

test("forwards model auth headers, retries transient errors, and maps usage", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeModel({
		api: "openai-responses",
		provider: "proxy",
		baseUrl: "https://provider.test/v1",
		headers: { "X-Model-Only": "present" },
	});
	const originalFetch = globalThis.fetch;
	let calls = 0;
	let request: { url: string; init: RequestInit } | undefined;
	let resolvedModel: Model<Api> | undefined;

	globalThis.fetch = async (url, init) => {
		calls++;
		request = { url: String(url), init };
		if (calls === 1) return new Response("temporary failure", { status: 503 });
		return new Response(
			JSON.stringify({
				status: "completed",
				output: [
					{ type: "web_search_call", id: "ws_1", action: { type: "search", query: "latest news" } },
					{
						type: "message",
						content: [{
							type: "output_text",
							text: "Search result.",
							annotations: [{
								type: "url_citation",
								url: "https://example.com/news",
								title: "Example News",
								start_index: 0,
								end_index: 14,
							}],
						}],
					},
				],
				usage: {
					input_tokens: 10,
					input_tokens_details: { cached_tokens: 2 },
					output_tokens: 20,
					output_tokens_details: { reasoning_tokens: 4 },
					total_tokens: 30,
				},
			}),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	};

	try {
		const result = await tool.execute("call_1", { query: "  latest news  " }, undefined, undefined, {
			model,
			modelRegistry: {
				getApiKeyAndHeaders: async (requestedModel: Model<Api>) => {
					resolvedModel = requestedModel;
					return {
						ok: true,
						apiKey: "secret",
						baseUrl: "https://provider.test/v1",
						headers: {
							"X-Provider": "provider",
							"X-Model-Only": requestedModel.headers?.["X-Model-Only"] ?? "",
						},
					};
				},
			},
		});

		assert.equal(calls, 2);
		assert.equal(resolvedModel, model);
		assert.equal(request?.url, "https://provider.test/v1/responses");
		const headers = request?.init.headers as Record<string, string>;
		assert.equal(headers["X-Provider"], "provider");
		assert.equal(headers["X-Model-Only"], "present");
		assert.equal(headers.Authorization, "Bearer secret");
		const body = JSON.parse(String(request?.init.body));
		assert.equal(body.input[0].content[0].text, "latest news");
		assert.deepEqual(body.reasoning, { effort: "low" });
		assert.equal(body.max_output_tokens, 8192);
		assert.match(result.content[0].text, /^Search result\./);
		assert.match(result.content[0].text, /Sources:\n- \[Example News\]\(<https:\/\/example\.com\/news>\)/);
		assert.deepEqual(result.details.searchCallIds, ["ws_1"]);
		assert.deepEqual(result.details.searchActions, [{ type: "search", query: "latest news" }]);
		assert.deepEqual(result.details.citations, [{
			url: "https://example.com/news",
			title: "Example News",
			startIndex: 0,
			endIndex: 14,
		}]);
		assert.equal(result.usage.input, 8);
		assert.equal(result.usage.output, 20);
		assert.equal(result.usage.cacheRead, 2);
		assert.equal(result.usage.reasoning, 4);
		assert.equal(result.usage.totalTokens, 30);
		assert.ok(result.usage.cost.total > 0);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("uses Pi's Codex transport and injects hosted web search", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeCodexModel();
	const accountId = "acct_test";
	const tokenPayload = Buffer.from(
		JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: accountId } }),
	).toString("base64url");
	const accessToken = `e30.${tokenPayload}.signature`;
	const originalFetch = globalThis.fetch;
	let request: { url: string; init: RequestInit } | undefined;
	let completeOptions: Record<string, unknown> | undefined;
	const sse = [
		{ type: "response.created", response: { id: "resp_search_1" } },
		{
			type: "response.output_item.done",
			output_index: 0,
			item: {
				type: "web_search_call",
				id: "ws_codex_1",
				action: {
					type: "search",
					query: "latest GPT news",
					sources: [
						{ type: "url", url: "https://example.com/gpt", title: "GPT News" },
						{ type: "url", url: "https://example.com/unreferenced", title: "Unreferenced source" },
						{ type: "url", url: "ftp://example.com/not-web", title: "Invalid source" },
					],
				},
			},
		},
		{
			type: "response.output_item.added",
			output_index: 0,
			item: { type: "message", id: "msg_search_1", content: [] },
		},
		{
			type: "response.output_item.done",
			output_index: 0,
			item: {
				type: "message",
				id: "msg_search_1",
				phase: "final_answer",
				content: [{
					type: "output_text",
					text: "Fresh web result.",
					annotations: [{
						type: "url_citation",
						url: "https://example.com/gpt",
						title: "GPT News",
						startIndex: 0,
						endIndex: 18,
					}],
				}],
			},
		},
		{
			type: "response.completed",
			response: {
				id: "resp_search_1",
				status: "completed",
				usage: { input_tokens: 12, output_tokens: 8, total_tokens: 20 },
			},
		},
	]
		.map((event, index) => {
			const json = JSON.stringify(event, null, index === 1 ? 2 : undefined);
			return index === 1
				? json.split("\n").map((line) => `data: ${line}`).join("\n") + "\n\n"
				: `data: ${json}\n\n`;
		})
		.join("");
	globalThis.fetch = async (url, init) => {
		request = { url: String(url), init };
		return new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } });
	};

	try {
		const result = await tool.execute("call_codex", { query: "latest GPT news" }, undefined, undefined, {
			model,
			modelRegistry: {
				complete: async (requestedModel: Model<Api>, context: any, options: Record<string, unknown>) => {
					completeOptions = options;
					return openAICodexResponsesApi().streamSimple(requestedModel, context, {
						...options,
						apiKey: accessToken,
						headers: { "X-Provider-Header": "present" },
					} as any).result();
				},
			},
		});

		assert.equal(request?.url, "https://chatgpt.com/backend-api/codex/responses");
		const headers = new Headers(request?.init.headers);
		assert.equal(headers.get("authorization"), `Bearer ${accessToken}`);
		assert.equal(headers.get("chatgpt-account-id"), accountId);
		assert.equal(headers.get("x-provider-header"), "present");
		const body = await decodeRequestBody(request?.init.body, headers.get("content-encoding") === "zstd");
		assert.deepEqual(body.tools, [{ type: "web_search" }]);
		assert.equal(body.tool_choice, "required");
		assert.ok(body.include.includes("web_search_call.action.sources"));
		assert.equal(body.max_output_tokens, undefined);
		assert.equal(body.reasoning.effort, "low");
		assert.equal(body.input[0].content[0].text, "latest GPT news");
		assert.match(result.content[0].text, /Fresh web result/);
		assert.match(result.content[0].text, /Sources:\n- \[GPT News\]\(<https:\/\/example\.com\/gpt>\)/);
		assert.doesNotMatch(result.content[0].text, /unreferenced|Invalid source/);
		assert.equal(result.details.model, "gpt-5.6-luna");
		assert.deepEqual(result.details.searchCallIds, ["ws_codex_1"]);
		assert.deepEqual(result.details.citations, [{
			url: "https://example.com/gpt",
			title: "GPT News",
			startIndex: 0,
			endIndex: 18,
		}]);
		assert.equal(completeOptions?.transport, "sse");
		assert.equal(completeOptions?.maxRetries, 1);
		assert.ok(completeOptions?.signal instanceof AbortSignal);
		assert.equal(result.usage.totalTokens, 20);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("passes caller cancellation through the Codex model registry request", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeCodexModel();
	const controller = new AbortController();
	let registrySignal: AbortSignal | undefined;

	const pending = tool.execute("call_codex_abort", { query: "latest GPT news" }, controller.signal, undefined, {
		model,
		modelRegistry: {
			complete: async (_requestedModel: Model<Api>, _context: any, options: { signal?: AbortSignal }) => {
				registrySignal = options.signal;
				await new Promise<never>((_resolve, reject) => {
					options.signal?.addEventListener("abort", () => reject(options.signal?.reason), { once: true });
				});
			},
		},
	});

	setTimeout(() => controller.abort(new Error("cancelled by caller")), 10);
	await assert.rejects(pending, /cancelled by caller/);
	assert.ok(registrySignal);
	assert.notEqual(registrySignal, controller.signal);
});

test("includes Codex search-call count when no readable result is returned", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];

	await assert.rejects(
		tool.execute("call_codex_empty", { query: "latest GPT news" }, undefined, undefined, {
			model: makeCodexModel(),
			modelRegistry: {
				complete: async () => ({ stopReason: "stop", content: [] } as any),
			},
		}),
		/search calls: none/,
	);
});

test("retries a transient network failure once", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeModel({ api: "openai-responses" });
	const originalFetch = globalThis.fetch;
	let calls = 0;
	globalThis.fetch = async () => {
		calls++;
		if (calls === 1) throw new Error("socket reset");
		return new Response(
			JSON.stringify({
				status: "completed",
				output: [{ type: "message", content: [{ type: "output_text", text: "Recovered result." }] }],
			}),
			{ status: 200 },
		);
	};

	try {
		const result = await tool.execute("call_1", { query: "query" }, undefined, undefined, {
			model,
			modelRegistry: {
				getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "secret", headers: {} }),
			},
		});
		assert.equal(calls, 2);
		assert.equal(result.content[0].text, "Recovered result.");
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("surfaces provider failure details and rejects empty queries", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeModel({ api: "openai-responses" });
	const context = {
		model,
		modelRegistry: {
			getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "secret", headers: {} }),
		},
	};
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () =>
		new Response(JSON.stringify({ status: "failed", error: { message: "search backend unavailable" } }), { status: 200 });

	try {
		await assert.rejects(
			tool.execute("call_1", { query: "query" }, undefined, undefined, context),
			/search backend unavailable/,
		);
		await assert.rejects(tool.execute("call_2", { query: "   " }, undefined, undefined, context), /must not be empty/);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("rejects unsupported models and unresolved credentials before fetching", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const originalFetch = globalThis.fetch;
	let calls = 0;
	globalThis.fetch = async () => {
		calls++;
		throw new Error("fetch should not be called");
	};

	try {
		await assert.rejects(
			tool.execute("call_1", { query: "query" }, undefined, undefined, {
				model: makeModel({ id: "other-model", api: "openai-completions" }),
				modelRegistry: {},
			}),
			/provider-side web search.*openai-codex\/gpt-5\.6-luna/,
		);
		await assert.rejects(
			tool.execute("call_2", { query: "query" }, undefined, undefined, {
				model: makeModel({ api: "openai-responses" }),
				modelRegistry: {
					getApiKeyAndHeaders: async () => ({ ok: false, error: "missing API key" }),
				},
			}),
			/missing API key/,
		);
		assert.equal(calls, 0);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("handles rejected, incomplete, queued, and malformed provider responses", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const context = {
		model: makeModel({ api: "openai-responses" }),
		modelRegistry: {
			getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "secret", headers: {} }),
		},
	};
	const responses = [
		new Response("null", { status: 400 }),
		new Response(JSON.stringify({ status: "incomplete", incomplete_details: { reason: "max_output_tokens" } }), {
			status: 200,
		}),
		new Response(JSON.stringify({ status: "queued" }), { status: 200 }),
		new Response("null", { status: 200 }),
	];
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () => responses.shift()!;

	try {
		await assert.rejects(tool.execute("call_1", { query: "query" }, undefined, undefined, context), /HTTP 400/);
		await assert.rejects(
			tool.execute("call_2", { query: "query" }, undefined, undefined, context),
			/reason: max_output_tokens/,
		);
		await assert.rejects(
			tool.execute("call_3", { query: "query" }, undefined, undefined, context),
			/status: queued/,
		);
		await assert.rejects(tool.execute("call_4", { query: "query" }, undefined, undefined, context), /no readable results/);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("propagates caller cancellation without retrying", async () => {
	const { tools } = setupExtension();
	const tool = tools[0];
	const model = makeModel({ api: "openai-responses" });
	const controller = new AbortController();
	controller.abort(new Error("cancelled by caller"));
	const originalFetch = globalThis.fetch;
	let calls = 0;
	globalThis.fetch = async (_url, init) => {
		calls++;
		assert.equal((init?.signal as AbortSignal).aborted, true);
		throw new Error("cancelled by caller");
	};

	try {
		await assert.rejects(
			tool.execute("call_1", { query: "query" }, controller.signal, undefined, {
				model,
				modelRegistry: {
					getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "secret", headers: {} }),
				},
			}),
			/cancelled by caller/,
		);
		assert.equal(calls, 1);
	} finally {
		globalThis.fetch = originalFetch;
	}
});
