/**
 * NVIDIA NIM API Provider Extension for pi
 *
 * Provides access to 100+ models from NVIDIA's NIM platform (build.nvidia.com)
 * via their OpenAI-compatible API endpoint.
 *
 * Setup:
 *   1. Get an API key from https://build.nvidia.com
 *   2. Export it: export NVIDIA_NIM_API_KEY=nvapi-... (or NVIDIA_API_KEY=nvapi-...)
 *   3. Load the extension:
 *      pi -e ./path/to/pi-nvidia-nim
 *      # or install as a package:
 *      pi install git:github.com/user/pi-nvidia-nim
 *
 * Then use /model and search for "nvidia-nim/" to see all available models.
 *
 * ## Reasoning / Thinking
 *
 * NVIDIA NIM models use `chat_template_kwargs` to enable thinking, which differs
 * from the standard OpenAI `reasoning_effort` parameter. This extension wraps the
 * standard streaming implementation and injects the correct per-model thinking
 * parameters:
 *
 * - DeepSeek V3.x: `chat_template_kwargs: { thinking: true }`
 * - DeepSeek V4:   `chat_template_kwargs: { thinking: true, reasoning_effort: "high" | "max" }`
 * - GLM-5/4.7:     `chat_template_kwargs: { enable_thinking: true, clear_thinking: false }`
 * - Kimi K2.5:     `chat_template_kwargs: { thinking: true }` (also accepts reasoning_effort)
 * - Qwen3:         `chat_template_kwargs: { enable_thinking: true }`
 *
 * NIM only accepts selected `reasoning_effort` values. The extension maps pi's
 * provider-agnostic levels to the values each NIM model accepts.
 *
 * Some models (e.g., GLM-5, GLM-4.7) always produce reasoning output regardless of
 * thinking settings.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
	ThinkingLevelMap,
} from "@earendil-works/pi-ai";
import { streamSimpleOpenAICompletions } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================

const NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1";
const NVIDIA_NIM_API_KEY_ENV = "NVIDIA_NIM_API_KEY";
const NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY";
const NVIDIA_API_KEY_ENV_NAMES = [NVIDIA_NIM_API_KEY_ENV, NVIDIA_API_KEY_ENV] as const;
const PROVIDER_NAME = "nvidia-nim";
const INKLING_MODEL_ID = "thinkingmachines/inkling";
const MINIMAX_M3_MODEL_ID = "minimaxai/minimax-m3";

// =============================================================================
// Per-model thinking configuration
// =============================================================================

/**
 * Maps model ID prefixes/exact IDs to their chat_template_kwargs for thinking.
 * When a user enables thinking in pi (any level > off), we inject these kwargs
 * into the request body. Models not listed here either:
 * - Don't support thinking (non-reasoning models)
 * - Always think regardless (GLM models without explicit kwargs)
 * - Work with standard reasoning_effort (rare on NIM)
 */
interface ThinkingConfig {
	/** chat_template_kwargs to send when thinking is enabled */
	enableKwargs: Record<string, unknown>;
	/** chat_template_kwargs to send when thinking is explicitly disabled (optional) */
	disableKwargs?: Record<string, unknown>;
	/** If true, also send reasoning_effort alongside chat_template_kwargs */
	sendReasoningEffort?: boolean;
	/** If true, include a model-specific reasoning_effort inside chat_template_kwargs */
	includeReasoningEffortInKwargs?: boolean;
}

const THINKING_CONFIGS: Record<string, ThinkingConfig> = {
	// DeepSeek models need chat_template_kwargs - reasoning_effort alone doesn't trigger thinking
	"deepseek-ai/deepseek-v4-flash": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		includeReasoningEffortInKwargs: true,
	},
	"deepseek-ai/deepseek-v4-pro": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		includeReasoningEffortInKwargs: true,
	},
	"deepseek-ai/deepseek-v3.2": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-v3.1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-v3.1-terminus": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-llama-8b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-7b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-14b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-32b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	// GLM models (Z-AI) - think by default, but can be controlled
	"z-ai/glm4.7": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	"z-ai/glm5": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	// Kimi models: chat_template_kwargs works, reasoning_effort also works
	"moonshotai/kimi-k2.6": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		sendReasoningEffort: true,
	},
	"moonshotai/kimi-k2-thinking": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		sendReasoningEffort: true,
	},
	// Qwen3 reasoning models
	"qwen/qwen3-235b-a22b": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwen3-coder-480b-a35b-instruct": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwen3-next-80b-a3b-thinking": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwq-32b": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	// Microsoft Phi reasoning
	"microsoft/phi-4-mini-flash-reasoning": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	// NVIDIA Nemotron reasoning models
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"nvidia/llama-3.3-nemotron-super-49b-v1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"nvidia/llama-3.3-nemotron-super-49b-v1.5": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	// Mistral reasoning
	"mistralai/magistral-small-2506": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
};

// =============================================================================
// Reasoning models and their capabilities
// =============================================================================

const REASONING_MODELS = new Set([
	...Object.keys(THINKING_CONFIGS),
	INKLING_MODEL_ID,
	MINIMAX_M3_MODEL_ID,
]);

// Models known to support image/vision input
const VISION_MODELS = new Set([
	"meta/llama-3.2-11b-vision-instruct",
	"meta/llama-3.2-90b-vision-instruct",
	"microsoft/phi-3-vision-128k-instruct",
	"microsoft/phi-3.5-vision-instruct",
	"microsoft/phi-4-multimodal-instruct",
	"nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
	"nvidia/nemotron-nano-12b-v2-vl",
	"nvidia/cosmos-reason2-8b",
	INKLING_MODEL_ID,
	MINIMAX_M3_MODEL_ID,
]);


// No presaved models: the list comes from the live /v1/models API at session start.
// No presaved model metadata. Discovery fetches the live NIM model list and enriches it
// with context/output/capability data from OpenRouter, then caches the result here so the
// provider exists at the next startup (pi needs >=1 model registered before session_start).
const MODEL_CACHE_PATH = join(homedir(), ".pi", "agent", "nvidia-nim-models.json");
const BOOTSTRAP_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b";
const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const NON_CHAT_PATTERN =
	/embed|reward|guard|safety|safeguard|parse|nvclip|clip\b|retriev|riva|vila|neva|deplot|kosmos|fuyu|paligemma|topic-control|streampetr|calibration|detector|translate/i;

function loadCachedModels(): NimModelEntry[] {
	try {
		if (existsSync(MODEL_CACHE_PATH)) {
			const models = JSON.parse(readFileSync(MODEL_CACHE_PATH, "utf8"));
			const valid = Array.isArray(models)
				? models.filter((m) => m && typeof m === "object" && typeof m.id === "string" && typeof m.contextWindow === "number")
				: [];
			if (valid.length > 0) return valid as NimModelEntry[];
		}
	} catch {}
	return [buildModelEntry(BOOTSTRAP_MODEL_ID)];
}

function saveCachedModels(models: NimModelEntry[]): void {
	try {
		mkdirSync(dirname(MODEL_CACHE_PATH), { recursive: true });
		writeFileSync(MODEL_CACHE_PATH, JSON.stringify(models));
	} catch {}
}

// -----------------------------------------------------------------------------
// OpenRouter metadata lookup
// -----------------------------------------------------------------------------

interface OpenRouterModel {
	id: string;
	context_length?: number;
	top_provider?: { max_completion_tokens?: number | null };
	architecture?: { input_modalities?: string[] };
	supported_parameters?: string[];
}

function modelSlug(id: string): string {
	return id.split("/").pop()!.toLowerCase().replace(/-(instruct|it|chat)$/, "");
}

/** Build a lookup that finds the best OpenRouter record for a NIM model id. */
function buildOpenRouterLookup(models: OpenRouterModel[]): (nimId: string) => OpenRouterModel | undefined {
	// Merge the plain and ":free" variants of each model, keeping whichever has the larger
	// context window. Other variants (":batch", "~alias") are ignored.
	const merged = new Map<string, OpenRouterModel>();
	for (const m of models) {
		if (m.id.startsWith("~") || (m.id.includes(":") && !m.id.endsWith(":free"))) continue;
		const key = m.id.replace(/:free$/, "");
		const current = merged.get(key);
		if (!current || (m.context_length ?? 0) > (current.context_length ?? 0)) merged.set(key, { ...m, id: key });
	}
	const base = Array.from(merged.values());
	const byId = new Map(base.map((m) => [m.id.toLowerCase(), m]));
	const bySlug = new Map<string, OpenRouterModel>();
	for (const m of base) if (!bySlug.has(modelSlug(m.id))) bySlug.set(modelSlug(m.id), m);
	return (nimId) => {
		const exact = byId.get(nimId.toLowerCase());
		if (exact) return exact;
		const slug = modelSlug(nimId);
		const sameSlug = bySlug.get(slug);
		if (sameSlug) return sameSlug;
		// OpenRouter often drops size suffixes (nemotron-3.5-lightning vs ...-30b-a3b):
		// accept the longest OpenRouter slug that is a prefix of the NIM slug at a "-" boundary.
		let best: OpenRouterModel | undefined;
		let bestLen = 0;
		for (const [orSlug, m] of bySlug) {
			if (slug.startsWith(`${orSlug}-`) && orSlug.length > bestLen && orSlug.length >= 8) {
				best = m;
				bestLen = orSlug.length;
			}
		}
		return best;
	};
}

async function fetchOpenRouterModels(): Promise<OpenRouterModel[]> {
	try {
		const response = await fetch(OPENROUTER_MODELS_URL, {
			headers: { Accept: "application/json" },
			signal: AbortSignal.timeout(15000),
		});
		if (!response.ok) return [];
		const data = (await response.json()) as { data?: OpenRouterModel[] };
		return Array.isArray(data.data) ? data.data : [];
	} catch {
		return [];
	}
}


// =============================================================================
// Custom streaming - wraps standard openai-completions with NIM-specific fixes
// =============================================================================

/**
 * Custom streamSimple that wraps the standard OpenAI completions streamer.
 *
 * Fixes for NVIDIA NIM:
 * 1. Maps pi's thinking levels to values accepted by NVIDIA NIM
 * 2. Strips reasoning_effort for models where it doesn't trigger thinking
 * 3. Injects chat_template_kwargs per model to actually enable thinking
 * 4. Uses onPayload callback to mutate request params before they're sent
 */
type NimApiKeyEnvName = (typeof NVIDIA_API_KEY_ENV_NAMES)[number];

function getNimApiKeyEnv(): NimApiKeyEnvName | undefined {
	return NVIDIA_API_KEY_ENV_NAMES.find((envName) => !!process.env[envName]);
}

function getNimApiKey(): string | undefined {
	const envName = getNimApiKeyEnv();
	const apiKey = envName ? process.env[envName] : undefined;
	return apiKey?.trim() || undefined;
}

function getNimProviderApiKeyConfig(): string {
	return `$${getNimApiKeyEnv() ?? NVIDIA_NIM_API_KEY_ENV}`;
}

function isNimApiKeyEnvName(value: string): value is NimApiKeyEnvName {
	return NVIDIA_API_KEY_ENV_NAMES.includes(value as NimApiKeyEnvName);
}

function isNimApiKeyEnvReference(value: string): boolean {
	return value.startsWith("$") && isNimApiKeyEnvName(value.slice(1));
}

function isNimApiKeyEnvPlaceholder(value: string): boolean {
	return isNimApiKeyEnvName(value) || isNimApiKeyEnvReference(value);
}

function resolveNimApiKeyEnvReference(value: string): string | undefined {
	if (!isNimApiKeyEnvReference(value)) return undefined;

	const envValue = process.env[value.slice(1)]?.trim();
	return envValue || undefined;
}

function normalizeResolvedNimApiKey(apiKey: string | undefined): string | undefined {
	if (apiKey === undefined) return undefined;

	const trimmed = apiKey.trim();
	if (!trimmed) {
		throw new Error("NVIDIA NIM API key resolved to an empty value.");
	}

	return trimmed;
}

function resolveNimApiKey(apiKey: string | undefined): string | undefined {
	const resolvedApiKey = normalizeResolvedNimApiKey(apiKey);

	if (resolvedApiKey) {
		const envReferenceApiKey = resolveNimApiKeyEnvReference(resolvedApiKey);
		if (envReferenceApiKey) return envReferenceApiKey;

		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(resolvedApiKey) && Object.hasOwn(process.env, resolvedApiKey)) {
			return normalizeResolvedNimApiKey(process.env[resolvedApiKey]);
		}

		if (!isNimApiKeyEnvPlaceholder(resolvedApiKey)) return resolvedApiKey;
	}

	return getNimApiKey();
}

function resolveRequiredNimApiKey(apiKey: string | undefined): string {
	const resolvedApiKey = resolveNimApiKey(apiKey);
	if (resolvedApiKey) return resolvedApiKey;

	throw new Error(
		`NVIDIA NIM: no API key configured. Set ${NVIDIA_NIM_API_KEY_ENV} or ${NVIDIA_API_KEY_ENV}. ` +
		`Get a free API key at https://build.nvidia.com and export it: ` +
		`export ${NVIDIA_NIM_API_KEY_ENV}=nvapi-...`,
	);
}

function mapNimTopLevelReasoning(reasoning: SimpleStreamOptions["reasoning"]): SimpleStreamOptions["reasoning"] {
	if (reasoning === "minimal") return "low";
	if (reasoning === "xhigh") return "high";
	return reasoning;
}

function mapDeepSeekV4Reasoning(reasoning: SimpleStreamOptions["reasoning"]): "high" | "max" {
	return reasoning === "xhigh" ? "max" : "high";
}

function buildThinkingKwargs(
	thinkingConfig: ThinkingConfig,
	reasoning: SimpleStreamOptions["reasoning"],
): Record<string, unknown> {
	const kwargs = { ...thinkingConfig.enableKwargs };
	if (thinkingConfig.includeReasoningEffortInKwargs) {
		kwargs.reasoning_effort = mapDeepSeekV4Reasoning(reasoning);
	}
	return kwargs;
}

function buildNimRequestHeaders(headers: SimpleStreamOptions["headers"], apiKey: string): Record<string, string> {
	const resolvedHeaders: Record<string, string> = {};

	for (const [key, value] of Object.entries(headers ?? {})) {
		if (key.toLowerCase() === "authorization" || value === null) continue;
		resolvedHeaders[key] = value;
	}

	return {
		...resolvedHeaders,
		Authorization: `Bearer ${apiKey}`,
	};
}

function nimStreamSimple(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	// pi-coding-agent registers streamSimple globally per `api` type (not per provider).
	// This streamer is invoked for ALL openai-completions providers (e.g. openrouter, openai),
	// not just nvidia-nim. Pass non-NIM calls through unchanged so we don't leak the
	// NVIDIA_NIM_API_KEY into other providers' Authorization headers.
	if (model.provider !== PROVIDER_NAME) {
		return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, options);
	}

	const thinkingConfig = THINKING_CONFIGS[model.id];
	const reasoning = options?.reasoning;
	const isThinkingEnabled = !!reasoning;

	// Custom NIM thinking configs use the provider-wide effort mapping. Models that
	// declare native chat-template compatibility keep their own named levels.
	const mappedReasoning = thinkingConfig ? mapNimTopLevelReasoning(reasoning) : reasoning;

	// For models that have a thinking config: we handle thinking via chat_template_kwargs.
	// Suppress reasoning_effort (set reasoning to undefined) unless the model explicitly
	// supports it alongside chat_template_kwargs (like Kimi).
	let effectiveReasoning = mappedReasoning;
	if (thinkingConfig && isThinkingEnabled && !thinkingConfig.sendReasoningEffort) {
		// Don't send reasoning_effort - we'll use chat_template_kwargs instead.
		// Setting to undefined prevents buildParams from adding reasoning_effort.
		effectiveReasoning = undefined;
	}

	// Use pi's already-resolved provider key when available (auth.json, shell command,
	// CLI override), and fall back to the two NVIDIA environment variable names.
	const nimApiKey = resolveRequiredNimApiKey(options?.apiKey);

	const modifiedOptions: SimpleStreamOptions = {
		...options,
		reasoning: effectiveReasoning,
		apiKey: nimApiKey,
		headers: buildNimRequestHeaders(options?.headers, nimApiKey),
		onPayload: (params: unknown) => {
			const p = params as Record<string, unknown>;

			if (thinkingConfig) {
				if (isThinkingEnabled) {
					// Inject chat_template_kwargs to enable thinking
					p.chat_template_kwargs = buildThinkingKwargs(thinkingConfig, reasoning);
				} else if (thinkingConfig.disableKwargs) {
					// Explicitly disable thinking (some models think by default, e.g. GLM-5/4.7)
					p.chat_template_kwargs = thinkingConfig.disableKwargs;
				}
			}

			// Ensure reasoning_effort is never "minimal" (belt & suspenders)
			if (p.reasoning_effort === "minimal") {
				p.reasoning_effort = "low";
			}

			// Normalize content arrays to plain strings where possible.
			// Many older/smaller NIM models (e.g., solar, baichuan, falcon) reject the
			// array format [{"type":"text","text":"..."}] and require a plain string.
			// This is safe for all models since plain strings are universally accepted.
			const messages = p.messages as Array<Record<string, unknown>> | undefined;
			if (messages) {
				for (const msg of messages) {
					if (Array.isArray(msg.content)) {
						const parts = msg.content as Array<Record<string, unknown>>;
						const allText = parts.every((part) => part.type === "text");
						if (allText) {
							msg.content = parts.map((part) => part.text as string).join("\n");
						}
					}
				}
			}

			// Chain to original onPayload if present
			return options?.onPayload?.(params, model);
		},
	};

	return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, modifiedOptions);
}

// =============================================================================
// Model building helpers
// =============================================================================

interface NimModelEntry {
	id: string;
	name: string;
	reasoning: boolean;
	thinkingLevelMap?: ThinkingLevelMap;
	input: ("text" | "image")[];
	contextWindow: number;
	maxTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	compat?: Record<string, unknown>;
}

function makeDisplayName(modelId: string): string {
	const parts = modelId.split("/");
	const name = parts[parts.length - 1];
	return name
		.replace(/-/g, " ")
		.replace(/_/g, " ")
		.replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildModelEntry(modelId: string, meta?: OpenRouterModel): NimModelEntry {
	const isReasoning = REASONING_MODELS.has(modelId) || !!meta?.supported_parameters?.includes("reasoning");
	const isVision = VISION_MODELS.has(modelId) || !!meta?.architecture?.input_modalities?.includes("image");
	// Unmatched models: honor a "-32k"/"-128k"/"8k" size hint in the id, else assume 128K.
	const idHint = modelId.match(/(?:^|[-_/])(\d+)k(?:$|[-_])/i);
	const contextWindow = meta?.context_length || (idHint ? Number(idHint[1]) * 1024 : 131072);
	const maxTokens = Math.min(meta?.top_provider?.max_completion_tokens || 16384, 32768, contextWindow);

	const entry: NimModelEntry = {
		id: modelId,
		name: makeDisplayName(modelId),
		reasoning: isReasoning,
		input: isVision ? ["text", "image"] : ["text"],
		contextWindow,
		maxTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	};

	// Default compat for all NIM models:
	// - supportsReasoningEffort: false - we handle thinking via streamSimple + chat_template_kwargs
	// - supportsDeveloperRole: false - "developer" role + chat_template_kwargs causes 500 on NIM
	//   (developer role alone works, but combined with thinking kwargs it breaks)
	// - maxTokensField: "max_tokens" - safer default for heterogeneous backends
	entry.compat = {
		supportsReasoningEffort: false,
		supportsDeveloperRole: false,
		maxTokensField: "max_tokens",
	};

	// Inkling conditions reasoning through its chat template rather than a top-level
	// reasoning_effort field. NVIDIA NIM accepts the same named effort presets as
	// Inkling's tokenizer; map pi's extended levels to Inkling's 0.99 preset.
	if (modelId === INKLING_MODEL_ID) {
		entry.thinkingLevelMap = { off: "none", xhigh: "max", max: "max" };
		entry.compat.thinkingFormat = "chat-template";
		entry.compat.chatTemplateKwargs = {
			reasoning_effort: { $var: "thinking.effort" },
		};
	}

	// MiniMax M3 exposes discrete disabled/adaptive/enabled reasoning modes through
	// chat_template_kwargs.thinking_mode. Map pi's effort scale to the closest mode.
	if (modelId === MINIMAX_M3_MODEL_ID) {
		entry.thinkingLevelMap = {
			off: "disabled",
			minimal: "adaptive",
			low: "adaptive",
			medium: "adaptive",
			high: "enabled",
			xhigh: "enabled",
			max: "enabled",
		};
		entry.compat.thinkingFormat = "chat-template";
		entry.compat.chatTemplateKwargs = {
			thinking_mode: { $var: "thinking.effort" },
		};
	}

	// Mistral models on NIM need extra compat flags
	if (modelId.startsWith("mistralai/")) {
		entry.compat.requiresToolResultName = true;
		entry.compat.requiresThinkingAsText = true;
		entry.compat.requiresMistralToolIds = true;
	}

	return entry;
}

// =============================================================================
// Dynamic model discovery
// =============================================================================

interface NimApiModel {
	id: string;
	object: string;
	owned_by: string;
}

type NimModelFetchResult =
	| { ok: true; modelIds: string[] }
	| { ok: false; reason: "auth" | "transient" | "invalid" | "network" | "other" };

const NIM_DISCOVERY_CREDENTIAL_WARNING =
	"NVIDIA NIM model discovery skipped: check your nvidia-nim credentials.";

function sanitizeNimLogMessage(message: string): string {
	return message.replace(/nvapi-[A-Za-z0-9._-]+/g, "nvapi-[REDACTED]");
}

function notifyNimDiscoveryCredentialWarning(ctx: ExtensionContext): void {
	ctx.ui.notify(NIM_DISCOVERY_CREDENTIAL_WARNING, "warning");
}

async function resolveNimDiscoveryApiKey(ctx: ExtensionContext): Promise<string | undefined> {
	try {
		const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_NAME);
		if (apiKey === undefined && ctx.modelRegistry.getProviderAuthStatus(PROVIDER_NAME).configured) {
			throw new Error("NVIDIA NIM configured credential resolved to an empty value.");
		}
		return resolveNimApiKey(apiKey);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		console.warn(`pi-nvidia-nim: ${sanitizeNimLogMessage(message)}`);
		notifyNimDiscoveryCredentialWarning(ctx);
		return undefined;
	}
}

async function fetchNimModels(apiKey: string): Promise<NimModelFetchResult> {
	try {
		const response = await fetch(`${NVIDIA_NIM_BASE_URL}/models`, {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				Accept: "application/json",
			},
			signal: AbortSignal.timeout(10000),
		});

		if (response.status === 401 || response.status === 403) {
			return { ok: false, reason: "auth" };
		}

		if (response.status === 429 || response.status >= 500) {
			return { ok: false, reason: "transient" };
		}

		if (!response.ok) return { ok: false, reason: "other" };

		const data = (await response.json()) as { data?: NimApiModel[] };
		if (!Array.isArray(data.data)) return { ok: false, reason: "invalid" };

		return {
			ok: true,
			modelIds: data.data.map((m) => m.id).filter((id): id is string => typeof id === "string" && id.length > 0),
		};
	} catch {
		return { ok: false, reason: "network" };
	}
}

// -----------------------------------------------------------------------------
// Liveness check
// -----------------------------------------------------------------------------
// NVIDIA's /v1/models list includes plenty of models that 410 (end-of-life) or
// 404 (not entitled on this account) on an actual chat completion. Before
// trusting a model, send it a minimal request and only keep it if NIM answers
// with 200.

const LIVENESS_CONCURRENCY = 8;
const LIVENESS_TIMEOUT_MS = 20000;
// NVIDIA's free tier returns 503 "Service temporarily overloaded" on some models
// fairly often even when the model is fine - a single retry isn't enough to tell
// genuinely dead models apart from ones that are just momentarily busy.
const LIVENESS_TRANSIENT_RETRIES = 4;
const LIVENESS_RETRY_DELAY_MS = 1500;
// Hard cap on the whole sweep, so a handful of slow/hanging models can't keep the
// pi process (and, in `-p` mode, its exit) lingering for minutes. Past this point
// in-flight requests are aborted and anything not yet confirmed alive is dropped
// for this round - it gets re-tested next session.
const LIVENESS_BUDGET_MS = 45000;

function delay(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
	const results: R[] = new Array(items.length);
	let next = 0;
	async function worker() {
		for (;;) {
			const i = next++;
			if (i >= items.length) return;
			results[i] = await fn(items[i]);
		}
	}
	await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
	return results;
}

async function isNimModelAlive(modelId: string, apiKey: string, budgetSignal: AbortSignal): Promise<boolean> {
	// Retry transient failures (429/5xx/network/timeout) several times with a short
	// delay - NVIDIA's free tier returns 503 on perfectly good models under load. A
	// 4xx other than 429 (400, 401, 403, 404, 410, ...) means the model itself is
	// unusable, so there's no point retrying that.
	for (let attempt = 0; attempt <= LIVENESS_TRANSIENT_RETRIES; attempt++) {
		if (budgetSignal.aborted) return false;
		try {
			const response = await fetch(`${NVIDIA_NIM_BASE_URL}/chat/completions`, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${apiKey}`,
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					model: modelId,
					messages: [{ role: "user", content: "hi" }],
					max_tokens: 1,
				}),
				// Whichever fires first: this request's own timeout, or the sweep's
				// overall time budget running out.
				signal: AbortSignal.any([budgetSignal, AbortSignal.timeout(LIVENESS_TIMEOUT_MS)]),
			});
			if (response.ok) return true;
			if (response.status !== 429 && response.status < 500) return false;
			// else: transient, fall through to retry
		} catch {
			// network error, timeout, or budget abort: fall through (the budget
			// check at the top of the next iteration catches the abort case)
		}
		if (attempt < LIVENESS_TRANSIENT_RETRIES && !budgetSignal.aborted) await delay(LIVENESS_RETRY_DELAY_MS);
	}
	return false;
}

/** Filters `models` down to the ones that actually answer a request on NIM right now. */
async function filterToLiveModels(models: NimModelEntry[], apiKey: string): Promise<NimModelEntry[]> {
	const budget = new AbortController();
	const budgetTimer = setTimeout(() => budget.abort(), LIVENESS_BUDGET_MS);
	try {
		const alive = await mapWithConcurrency(models, LIVENESS_CONCURRENCY, (m) =>
			isNimModelAlive(m.id, apiKey, budget.signal),
		);
		return models.filter((_, i) => alive[i]);
	} finally {
		clearTimeout(budgetTimer);
	}
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
	const providerApiKeyConfig = getNimProviderApiKeyConfig();

	// Register cached models immediately. The request path resolves credentials through
	// pi first (CLI override, auth.json, shell command), then falls back to
	// NVIDIA_NIM_API_KEY/NVIDIA_API_KEY.
	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: NVIDIA_NIM_BASE_URL,
		apiKey: providerApiKeyConfig,
		api: "openai-completions",
		authHeader: true,
		models: loadCachedModels(),
		streamSimple: nimStreamSimple,
	});

	// On session start, refresh the model list from NIM and metadata from OpenRouter.
	pi.on("session_start", async (_event, ctx) => {
		const apiKey = await resolveNimDiscoveryApiKey(ctx);
		if (!apiKey) return;

		const [fetchResult, openRouterModels] = await Promise.all([fetchNimModels(apiKey), fetchOpenRouterModels()]);
		if (!fetchResult.ok) {
			if (fetchResult.reason === "auth") {
				notifyNimDiscoveryCredentialWarning(ctx);
			}
			return;
		}

		const chatIds = fetchResult.modelIds.filter((id) => !NON_CHAT_PATTERN.test(id));
		if (chatIds.length === 0) return;

		const lookup = buildOpenRouterLookup(openRouterModels);
		// If OpenRouter is unreachable, keep whatever metadata we cached earlier.
		const previous = new Map(loadCachedModels().map((m) => [m.id, m]));
		const candidates = chatIds.map((id) => {
			const meta = lookup(id);
			if (!meta && openRouterModels.length === 0 && previous.has(id)) return previous.get(id)!;
			return buildModelEntry(id, meta);
		});

		// Drop anything that doesn't actually answer right now (EOL/410, not
		// entitled/404, etc.). This sweep can take a while across many candidates
		// (retries on top of per-request timeouts), so it must not hold up pi's
		// startup - don't await it here. pi already has cached/previous models
		// registered above; swap in the validated list whenever the sweep finishes.
		void filterToLiveModels(candidates, apiKey).then((models) => {
			// If every single model fails the check - e.g. the key just got
			// revoked - that's more likely a systemic problem than every model
			// going down at once, so keep the existing cache instead of wiping it.
			if (models.length === 0) return;

			saveCachedModels(models);
			try {
				// The sweep can easily outlive a short-lived (-p) session, or run
				// past a session reload/fork/switch in an interactive one. pi's
				// extension ctx throws if used after that, which would otherwise
				// surface as an unhandled rejection - there's simply nothing left
				// to register against, so just drop the update.
				pi.registerProvider(PROVIDER_NAME, {
					baseUrl: NVIDIA_NIM_BASE_URL,
					apiKey: getNimProviderApiKeyConfig(),
					api: "openai-completions",
					authHeader: true,
					models,
					streamSimple: nimStreamSimple,
				});
			} catch {
				// Session ended or was replaced before the sweep finished - nothing to update.
			}
		});
	});
}
