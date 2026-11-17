// free-model-aggregator-02 entry: a best-first free-model aggregator.
//
// One server speaking both client wires (OpenAI Chat Completions and Anthropic Messages, JSON and
// streaming). Every prompt is dialled, through the trial gateway's Chat Completions route, to the
// free OpenRouter model with the highest expected quality for that prompt's requirements, and the
// answer is returned in the client's own wire with the dialled model's id on it.
//
// Routing, in the order a prompt is served:
//   - candidates are the brief's free models that meet the request's requirements (JSON mode,
//     tools, vision), best first by expected quality: the published rating discounted shallowly by
//     a Bayesian success estimate, rating * ((1 - w) + w * P), P from a Beta(27, 5) prior updated
//     with every gateway outcome this trial has shown us. The reliability weight w starts at the
//     shallow 0.08 and grows toward 0.20 as the trial's call budget is spent, so late in a trial a
//     proven model outranks a merely famous one: risk is priced against the calls and prompts that
//     remain, never against a clock, which keeps every choice a function of the brief, the requests
//     and the gateway's answers, and a replay therefore makes the same calls in the same order.
//   - a refusal is retried once on the same model when it looks transient (429, 5xx, a timeout)
//     and it is the prompt's best candidate, so a rate limit costs one extra call instead of the
//     prompt's best model; then the prompt fails over to the next candidate, and the next, up to
//     five calls a prompt. A model that answers with no trace of the token the request asked for
//     is not asked again — the identical ask would draw the identical answer — the prompt fails
//     over instead. A 402 or 404 retires the model for the trial, a gateway-level refusal (401,
//     the request cap, the budget) retires the gateway, and every refusal met on the way is
//     reported on the answer's x-makefun-upstream-failures trail.
//   - an answer counts only when its text carries the token the prompt asked for; anything else is
//     treated as a failure of that model for that prompt and the next candidate is tried.
//
// The server registers its catalog with the env, polls /work, and exits 0 when the env is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small shapes

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function strOr(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function log(message: string): void {
  process.stderr.write(`aggregator-02: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// The brief: the free model list (and the prompt manifest, unused for routing:
// requirements are read off each request body, which is where they bind).

interface ModelInfo {
  /** The dialable id the gateway routes, openrouter/<vendor>/<model>:free. */
  id: string;
  displayName: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  /** Position in the brief, the final deterministic tie-break. */
  order: number;
}

/** The JSON document in `text` whose opening is `{"schema":"<schema>"`, brace-matched whole. */
function extractDocument(text: string, schema: string): Json | null {
  const at = text.indexOf(`{"schema":"${schema}"`);
  if (at === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = at; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          const value: unknown = JSON.parse(text.slice(at, i + 1));
          return isRecord(value) ? value : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function readBriefModels(): ModelInfo[] {
  const file = process.env.MAKEFUN_PROMPT_FILE ?? '';
  if (file === '') return [];
  let text: string;
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return [];
  }
  const doc = extractDocument(text, 'free-model-aggregator-02/models/v1');
  if (doc === null) return [];
  const models: ModelInfo[] = [];
  for (const raw of asArray(doc.models)) {
    if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id === '') continue;
    if (raw.free === false) continue;
    models.push({
      id: raw.id,
      displayName: strOr(raw.display_name, raw.id),
      provider: strOr(raw.provider, 'openrouter'),
      ratingMilli: numOr(raw.rating_milli, 0),
      contextLength: typeof raw.context_length === 'number' ? raw.context_length : null,
      tools: raw.tools === true,
      vision: raw.vision === true,
      json: raw.json === true,
      order: models.length,
    });
  }
  return models;
}

const MODELS: ModelInfo[] = readBriefModels();

// ---------------------------------------------------------------------------
// The catalog: one provider, the brief's free models, the same document in the
// register body and on --list.

interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

function buildCatalog(): Catalog {
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: MODELS.map((m) => ({ id: m.id, free: true, context_length: m.contextLength })),
      },
    ],
  };
}

const CATALOG: Catalog = buildCatalog();
const CATALOG_TEXT = `${JSON.stringify(CATALOG)}\n`;

// ---------------------------------------------------------------------------
// Routing state: everything here is a function of the gateway's answers so
// far, so a replay walks the same path.

interface ModelState {
  successes: number;
  failures: number;
  retired: boolean;
}

const modelState = new Map<string, ModelState>();
let callsMade = 0;
let gatewayRetired = false;

function stateOf(model: ModelInfo): ModelState {
  let s = modelState.get(model.id);
  if (s === undefined) {
    s = { successes: 0, failures: 0, retired: false };
    modelState.set(model.id, s);
  }
  return s;
}

/** The Beta(27, 5) posterior mean of the model's success probability. */
function successEstimate(model: ModelInfo): number {
  const s = stateOf(model);
  return (27 + s.successes) / (32 + s.successes + s.failures);
}

/** How much of the ordering the success estimate may move: shallow early, firmer as calls go. */
function reliabilityWeight(): number {
  return 0.08 + 0.12 * Math.min(1, callsMade / 30);
}

function expectedQuality(model: ModelInfo): number {
  const w = reliabilityWeight();
  return model.ratingMilli * ((1 - w) + w * successEstimate(model));
}

interface Requirements {
  json: boolean;
  tools: boolean;
  vision: boolean;
}

function meets(model: ModelInfo, req: Requirements): boolean {
  if (req.json && !model.json) return false;
  if (req.tools && !model.tools) return false;
  if (req.vision && !model.vision) return false;
  return true;
}

/** The candidates for a prompt, best expected quality first. */
function candidatesFor(req: Requirements): ModelInfo[] {
  return MODELS.filter((m) => !stateOf(m).retired && meets(m, req)).sort(
    (a, b) => expectedQuality(b) - expectedQuality(a) || b.ratingMilli - a.ratingMilli || a.order - b.order,
  );
}

// ---------------------------------------------------------------------------
// Requests: what a prompt needs, and the token it asks for.

/** The content parts of a Chat Completions message, as plain text. */
function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('\n');
}

function requirementsOfGatewayBody(body: Json): Requirements {
  let vision = false;
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const part of raw.content) {
      if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    json: format === 'json_object' || format === 'json_schema',
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
  };
}

const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/;

/** The token the prompt's instruction asks to have echoed back, or null when it names none. */
function expectedToken(body: Json): string | null {
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    const found = TOKEN_PATTERN.exec(messageText(raw.content));
    if (found !== null) return found[0];
  }
  return null;
}

/** A rough deterministic token estimate, used only when a provider omits usage entirely. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

// ---------------------------------------------------------------------------
// Wire translation: Anthropic Messages <-> Chat Completions.

/** An Anthropic system field (a string or text blocks) as one string. */
function systemText(system: unknown): string {
  if (typeof system === 'string') return system;
  if (!Array.isArray(system)) return '';
  return system
    .map((block) => (isRecord(block) && typeof block.text === 'string' ? block.text : ''))
    .filter((t) => t !== '')
    .join('\n');
}

/** One Anthropic image block as an OpenAI image_url part. */
function imagePart(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as the text of a Chat Completions tool message. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = content
      .map((block) => (isRecord(block) && typeof block.text === 'string' ? block.text : ''))
      .join('');
    if (text !== '') return text;
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages body as the Chat Completions body this entry dials (model set per attempt). */
function anthropicToChat(body: Json): Json {
  const messages: Json[] = [];
  const system = systemText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      const texts: string[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
        else if (block.type === 'tool_use') {
          toolCalls.push({
            id: strOr(block.id, ''),
            type: 'function',
            function: {
              name: strOr(block.name, ''),
              arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
            },
          });
        }
      }
      const message: Json = { role, content: texts.length > 0 ? texts.join('') : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    const parts: Json[] = [];
    const toolMessages: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        toolMessages.push({
          role: 'tool',
          tool_call_id: strOr(block.tool_use_id, ''),
          content: toolResultText(block.content),
        });
      } else if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image') {
        const part = imagePart(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length === 1 && parts[0].type === 'text') messages.push({ role, content: parts[0].text });
    else if (parts.length > 0) messages.push({ role, content: parts });
    for (const toolMessage of toolMessages) messages.push(toolMessage);
  }

  const out: Json = { messages };
  out.max_tokens = clampCap(body.max_tokens);
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools
      .filter((raw) => isRecord(raw) && typeof raw.name === 'string')
      .map((raw) => {
        const tool = raw as Json;
        return {
          type: 'function',
          function: {
            name: tool.name,
            description: strOr(tool.description, ''),
            parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
          },
        };
      });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  return out;
}

/** The gateway's output cap for a body: the client's own when it names a legal one, else a default. */
function clampCap(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 1) return 1024;
  return Math.min(8192, Math.floor(value));
}

/** An OpenAI-shaped client body as the gateway body: the client's fields, the cap normalised. */
function openAiGatewayBody(body: Json): Json {
  const out: Json = { ...body };
  delete out.model;
  delete out.stream;
  delete out.stream_options;
  const cap = typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : out.max_tokens;
  const clamped = clampCap(cap);
  if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = clamped;
  else out.max_tokens = clamped;
  return out;
}

// ---------------------------------------------------------------------------
// The gateway: one dial, its assembled answer, or its refusal.

interface ToolCall {
  id: string;
  name: string;
  args: string;
}

interface Completion {
  text: string;
  toolCalls: ToolCall[];
  finishReason: string | null;
  usage: { prompt: number; completion: number } | null;
  id: string | null;
  created: number | null;
  /** The provider's whole JSON answer, on a non-streamed call. */
  json: Json | null;
}

type DialResult =
  | { kind: 'ok'; completion: Completion }
  | { kind: 'refused'; status: number; code: string };

const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
const ATTEMPT_TIMEOUT_MS = 9000;

function chatCompletionsUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

/** The error code a refusal body names: error.code, else error.type, else ''. */
function refusalCode(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      if (typeof code === 'number') return String(code);
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to name.
  }
  return '';
}

/** A non-streamed Chat Completions answer as a Completion. */
function completionFromJson(body: Json): Completion {
  const choices = asArray(body.choices);
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ToolCall[] = [];
  for (const raw of asArray(message.tool_calls)) {
    if (!isRecord(raw) || !isRecord(raw.function)) continue;
    toolCalls.push({
      id: strOr(raw.id, ''),
      name: strOr(raw.function.name, ''),
      args: strOr(raw.function.arguments, ''),
    });
  }
  const usage = isRecord(body.usage)
    ? { prompt: numOr(body.usage.prompt_tokens, 0), completion: numOr(body.usage.completion_tokens, 0) }
    : null;
  return {
    text: messageText(message.content),
    toolCalls,
    finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    usage,
    id: typeof body.id === 'string' ? body.id : null,
    created: typeof body.created === 'number' ? body.created : null,
    json: body,
  };
}

/** A streamed Chat Completions answer (the whole SSE text) as a Completion. */
function completionFromStream(text: string): Completion | null {
  let assembled = '';
  const toolByIndex = new Map<number, ToolCall>();
  let finishReason: string | null = null;
  let usage: { prompt: number; completion: number } | null = null;
  let id: string | null = null;
  let created: number | null = null;
  let sawChunk = false;
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice('data:'.length).trim();
    if (payload === '' || payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawChunk = true;
    if (typeof chunk.id === 'string') id = chunk.id;
    if (typeof chunk.created === 'number') created = chunk.created;
    if (isRecord(chunk.usage)) {
      usage = { prompt: numOr(chunk.usage.prompt_tokens, 0), completion: numOr(chunk.usage.completion_tokens, 0) };
    }
    const choices = asArray(chunk.choices);
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') assembled += delta.content;
    for (const raw of asArray(delta.tool_calls)) {
      if (!isRecord(raw)) continue;
      const index = typeof raw.index === 'number' ? raw.index : toolByIndex.size;
      let call = toolByIndex.get(index);
      if (call === undefined) {
        call = { id: '', name: '', args: '' };
        toolByIndex.set(index, call);
      }
      if (typeof raw.id === 'string' && raw.id !== '') call.id = raw.id;
      if (isRecord(raw.function)) {
        if (typeof raw.function.name === 'string' && raw.function.name !== '') call.name = raw.function.name;
        if (typeof raw.function.arguments === 'string') call.args += raw.function.arguments;
      }
    }
    if (typeof choice.finish_reason === 'string') finishReason = choice.finish_reason;
  }
  if (!sawChunk) return null;
  const toolCalls = [...toolByIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  return { text: assembled, toolCalls, finishReason, usage, id, created, json: null };
}

/** One call to the gateway's Chat Completions route for one model. */
async function dial(model: ModelInfo, gatewayBody: Json, stream: boolean): Promise<DialResult> {
  const body: Json = { ...gatewayBody, model: model.id };
  if (stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(stream ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
    });
  } catch {
    // A timeout or a broken connection: no status came back; 504 names it for the trail.
    return { kind: 'refused', status: 504, code: 'gateway_timeout' };
  }
  const text = await response.text();
  if (!response.ok) return { kind: 'refused', status: response.status, code: refusalCode(text) };
  if (stream) {
    const completion = completionFromStream(text);
    if (completion === null) return { kind: 'refused', status: 502, code: 'empty_stream' };
    return { kind: 'ok', completion };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { kind: 'refused', status: 502, code: 'invalid_upstream_json' };
  }
  if (!isRecord(parsed)) return { kind: 'refused', status: 502, code: 'invalid_upstream_json' };
  if (isRecord(parsed.error) && !Array.isArray(parsed.choices)) {
    // Some upstreams answer 200 with an error body: it is a refusal wearing a success status.
    const code = parsed.error.code;
    const status = typeof code === 'number' && code >= 100 && code <= 599 ? code : 502;
    return { kind: 'refused', status, code: refusalCode(text) || 'upstream_error' };
  }
  return { kind: 'ok', completion: completionFromJson(parsed) };
}

// ---------------------------------------------------------------------------
// Answers: the completion in the client's wire, the dialled model's id on it.

const TRIAL_NOW = process.env.MAKEFUN_TRIAL_NOW ?? '';
const FALLBACK_CREATED = (() => {
  const parsed = Date.parse(TRIAL_NOW);
  return Number.isNaN(parsed) ? 1735689600 : Math.floor(parsed / 1000);
})();

type Wire = 'openai' | 'anthropic';

interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

function trailHeader(trail: TrailRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, 8)) };
}

function sendBody(res: ServerResponse, status: number, contentType: string, text: string, extra: Record<string, string>): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendJson(res: ServerResponse, status: number, value: unknown, extra: Record<string, string> = {}): void {
  sendBody(res, status, 'application/json', JSON.stringify(value), extra);
}

/** A stop reason the OpenAI wire knows, from a completion that may name none. */
function openAiFinish(completion: Completion): string {
  if (completion.finishReason !== null && completion.finishReason !== '') return completion.finishReason;
  return completion.toolCalls.length > 0 ? 'tool_calls' : 'stop';
}

/** A completion's usage, the provider's own when it sent one, an estimate when it sent none. */
function usageOf(completion: Completion, requestText: string): { prompt: number; completion: number } {
  if (completion.usage !== null) {
    return {
      prompt: Math.max(0, Math.floor(completion.usage.prompt)),
      completion: Math.max(1, Math.floor(completion.usage.completion)),
    };
  }
  return { prompt: estimateTokens(requestText), completion: estimateTokens(completion.text) };
}

/** The OpenAI JSON answer: the provider's own body, the dialled id on it, its gaps closed. */
function openAiJsonAnswer(completion: Completion, model: ModelInfo, requestText: string): Json {
  const usage = usageOf(completion, requestText);
  const base: Json = completion.json !== null ? { ...completion.json } : {};
  const choices = asArray(base.choices);
  const choice: Json = isRecord(choices[0]) ? { ...choices[0] } : {};
  if (!isRecord(choice.message)) {
    choice.message = {
      role: 'assistant',
      content: completion.text,
      ...(completion.toolCalls.length > 0
        ? {
            tool_calls: completion.toolCalls.map((call) => ({
              id: call.id,
              type: 'function',
              function: { name: call.name, arguments: call.args },
            })),
          }
        : {}),
    };
  }
  choice.finish_reason = openAiFinish(completion);
  base.choices = [choice, ...choices.slice(1)];
  base.model = model.id;
  base.object = 'chat.completion';
  if (typeof base.id !== 'string') base.id = completion.id ?? 'chatcmpl-aggregator';
  if (typeof base.created !== 'number') base.created = completion.created ?? FALLBACK_CREATED;
  base.usage = {
    prompt_tokens: usage.prompt,
    completion_tokens: usage.completion,
    total_tokens: usage.prompt + usage.completion,
  };
  return base;
}

/** One SSE event frame. */
function sse(data: unknown, event: string | null): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The OpenAI SSE answer, rebuilt from the assembled completion: content, finish, usage, [DONE]. */
function openAiStreamAnswer(completion: Completion, model: ModelInfo, requestText: string): string {
  const usage = usageOf(completion, requestText);
  const id = completion.id ?? 'chatcmpl-aggregator';
  const created = completion.created ?? FALLBACK_CREATED;
  const chunk = (choices: unknown[], withUsage: boolean): Json => ({
    id,
    object: 'chat.completion.chunk',
    created,
    model: model.id,
    choices,
    ...(withUsage
      ? { usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion } }
      : {}),
  });
  let out = '';
  if (completion.text !== '') {
    out += sse(chunk([{ index: 0, delta: { role: 'assistant', content: completion.text }, finish_reason: null }], false), null);
  }
  completion.toolCalls.forEach((call, index) => {
    out += sse(
      chunk(
        [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index, id: call.id !== '' ? call.id : `call_${index}`, type: 'function', function: { name: call.name, arguments: call.args } },
              ],
            },
            finish_reason: null,
          },
        ],
        false,
      ),
      null,
    );
  });
  if (completion.text === '' && completion.toolCalls.length === 0) {
    out += sse(chunk([{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }], false), null);
  }
  out += sse(chunk([{ index: 0, delta: {}, finish_reason: openAiFinish(completion) }], false), null);
  out += sse(chunk([], true), null);
  out += 'data: [DONE]\n\n';
  return out;
}

/** Chat Completions' finish_reason as Anthropic's stop_reason. */
function anthropicStopReason(completion: Completion): string {
  switch (completion.finishReason) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return completion.toolCalls.length > 0 ? 'tool_use' : 'end_turn';
  }
}

/** A tool call's arguments string as the input object of an Anthropic tool_use block. */
function argumentsObject(args: string): Json {
  if (args.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(args);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The Anthropic JSON answer: text and tool_use blocks, the stop reason and usage translated. */
function anthropicJsonAnswer(completion: Completion, model: ModelInfo, requestText: string): Json {
  const usage = usageOf(completion, requestText);
  const content: Json[] = [];
  if (completion.text !== '') content.push({ type: 'text', text: completion.text });
  completion.toolCalls.forEach((call, index) => {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : `toolu_${index}`,
      name: call.name,
      input: argumentsObject(call.args),
    });
  });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: completion.id !== null ? `msg_${completion.id}` : 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: model.id,
    content,
    stop_reason: anthropicStopReason(completion),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  };
}

/** The Anthropic SSE answer: the canonical event order, rebuilt from the assembled completion. */
function anthropicStreamAnswer(completion: Completion, model: ModelInfo, requestText: string): string {
  const usage = usageOf(completion, requestText);
  let out = sse(
    {
      type: 'message_start',
      message: {
        id: completion.id !== null ? `msg_${completion.id}` : 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model: model.id,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt, output_tokens: 0 },
      },
    },
    'message_start',
  );
  let index = 0;
  if (completion.text !== '') {
    out += sse({ type: 'content_block_start', index, content_block: { type: 'text', text: '' } }, 'content_block_start');
    out += sse(
      { type: 'content_block_delta', index, delta: { type: 'text_delta', text: completion.text } },
      'content_block_delta',
    );
    out += sse({ type: 'content_block_stop', index }, 'content_block_stop');
    index += 1;
  }
  for (const call of completion.toolCalls) {
    out += sse(
      {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      },
      'content_block_start',
    );
    if (call.args !== '') {
      out += sse(
        { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } },
        'content_block_delta',
      );
    }
    out += sse({ type: 'content_block_stop', index }, 'content_block_stop');
    index += 1;
  }
  if (index === 0) {
    out += sse({ type: 'content_block_start', index, content_block: { type: 'text', text: '' } }, 'content_block_start');
    out += sse({ type: 'content_block_stop', index }, 'content_block_stop');
  }
  out += sse(
    {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(completion), stop_sequence: null },
      usage: { output_tokens: usage.completion },
    },
    'message_delta',
  );
  out += sse({ type: 'message_stop' }, 'message_stop');
  return out;
}

/** The answer that gives a prompt up, in the client's wire, with the trail of what was tried. */
function giveUp(res: ServerResponse, wire: Wire, trail: TrailRow[], message: string): void {
  if (wire === 'anthropic') {
    sendJson(res, 502, { type: 'error', error: { type: 'api_error', message } }, trailHeader(trail));
  } else {
    sendJson(res, 502, { error: { message, type: 'upstream_error', code: 'upstream_unavailable' } }, trailHeader(trail));
  }
}

// ---------------------------------------------------------------------------
// Serving one prompt: candidates in expected-quality order, the retry and
// failover policy, and the answer in the client's wire.

const MAX_ATTEMPTS_PER_PROMPT = 5;
const RETRY_WAIT_MS = 700;
/** Gateway codes that mean the trial's lane itself is closed, not one model. */
const GATEWAY_CODES = new Set(['request_cap', 'budget_exhausted']);

async function serve(
  res: ServerResponse,
  wire: Wire,
  clientBody: Json,
  gatewayBody: Json,
  stream: boolean,
): Promise<void> {
  const requirements = requirementsOfGatewayBody(gatewayBody);
  const token = expectedToken(gatewayBody);
  const requestText = asArray(gatewayBody.messages)
    .map((raw) => (isRecord(raw) ? messageText(raw.content) : ''))
    .join('\n');
  const trail: TrailRow[] = [];

  if (gatewayRetired) {
    giveUp(res, wire, trail, 'the gateway refused this trial’s earlier calls; no model can be dialled');
    return;
  }

  let attempts = 0;
  const candidates = candidatesFor(requirements);
  for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    const candidate = candidates[candidateIndex];
    const state = stateOf(candidate);
    let triesOnCandidate = 0;
    while (attempts < MAX_ATTEMPTS_PER_PROMPT) {
      attempts += 1;
      triesOnCandidate += 1;
      callsMade += 1;
      const result = await dial(candidate, gatewayBody, stream);
      if (result.kind === 'ok') {
        if (token !== null && !result.completion.text.includes(token)) {
          // The model answered but not the answer asked for: the identical ask would draw the
          // identical answer, so the prompt fails over to the next candidate instead.
          state.failures += 1;
          break;
        }
        state.successes += 1;
        const extra = trailHeader(trail);
        if (wire === 'openai') {
          if (stream) sendBody(res, 200, 'text/event-stream', openAiStreamAnswer(result.completion, candidate, requestText), extra);
          else sendJson(res, 200, openAiJsonAnswer(result.completion, candidate, requestText), extra);
        } else {
          if (stream) sendBody(res, 200, 'text/event-stream', anthropicStreamAnswer(result.completion, candidate, requestText), extra);
          else sendJson(res, 200, anthropicJsonAnswer(result.completion, candidate, requestText), extra);
        }
        return;
      }
      // A refusal: record it for the trail, then classify it.
      trail.push({ status: result.status, code: result.code, model: candidate.id });
      if (result.status === 401 || GATEWAY_CODES.has(result.code)) {
        gatewayRetired = true;
        giveUp(res, wire, trail, 'the gateway refused the trial’s key or its call budget');
        return;
      }
      if (result.status === 402 || result.status === 404) {
        // This model will not serve on this lane: retire it and fail over.
        state.retired = true;
        state.failures += 1;
        break;
      }
      if (result.status === 429 || result.status >= 500) {
        // Transient: the best candidate earns one same-model retry (the retry the failover
        // column rewards); later candidates get the one call, then the prompt moves on.
        state.failures += 1;
        if (candidateIndex === 0 && triesOnCandidate < 2 && attempts < MAX_ATTEMPTS_PER_PROMPT) {
          await sleep(RETRY_WAIT_MS);
          continue;
        }
        break;
      }
      // Any other 4xx is this request meeting this model: demote it and fail over, no retry.
      state.failures += 2;
      break;
    }
  }
  giveUp(res, wire, trail, 'every free model that meets this request refused or failed it');
}

// ---------------------------------------------------------------------------
// The HTTP surface.

const MAX_BODY_BYTES = 8 * 1024 * 1024;

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function wireError(res: ServerResponse, wire: Wire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: status === 404 ? 'not_found' : type } });
}

/** GET /v1/models: the OpenAI list for a Bearer header, the Anthropic list for an x-api-key. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: MODELS.map((m) => ({
        type: 'model',
        id: m.id,
        display_name: m.displayName,
        created_at: TRIAL_NOW !== '' ? TRIAL_NOW : '2025-01-01T00:00:00Z',
      })),
      has_more: false,
      first_id: MODELS.length > 0 ? MODELS[0].id : null,
      last_id: MODELS.length > 0 ? MODELS[MODELS.length - 1].id : null,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: MODELS.map((m) => ({ id: m.id, object: 'model', created: FALLBACK_CREATED, owned_by: m.provider })),
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJsonBody(req);
    if (body === null) return wireError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'openai', body, openAiGatewayBody(body), body.stream === true);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJsonBody(req);
    if (body === null) return wireError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'anthropic', body, anthropicToChat(body), body.stream === true);
    return;
  }
  wireError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// The trial lifecycle: listen, register, poll /work, exit 0.

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const POLL_INTERVAL_MS = 150;

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register until the env takes it: it may still be starting when we are. */
async function register(port: number): Promise<void> {
  const body = JSON.stringify({ protocol_version: 1, port, catalog: CATALOG });
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const reply: unknown = await res.json();
        if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} prompts`);
        return;
      }
      log(`register answered HTTP ${res.status}; retrying`);
    } catch {
      // The env is not up yet.
    }
    await sleep(500);
  }
  throw new Error('the env never accepted the register call');
}

/** Poll GET /work; exit 0 the moment the env says every prompt has been sent. */
function pollWork(): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            process.exit(0);
          }
        }
      } catch {
        // Not reachable yet: the next tick tries again.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(CATALOG_TEXT, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    // A client that stopped waiting (its own 30 s cap) must never take the process down.
    res.on('error', () => {});
    void handle(req, res).catch((err: unknown) => {
      log(err instanceof Error ? (err.stack ?? err.message) : String(err));
      if (!res.headersSent) {
        wireError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      } else if (!res.writableEnded) {
        res.end();
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns: restart');
  log(`listening on 0.0.0.0:${address.port} with ${MODELS.length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork();
}

try {
  await start();
} catch (err) {
  log(err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
}
