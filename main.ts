// A free-model aggregator for the make.fun trial.
//
// The directory's providers are simulated: each one echoes the request it was
// given (system text, user text, tool names, max_tokens, stops, image bytes, JSON
// mode), enforces a per-trial budget of successful calls, and sometimes refuses a
// call with a 402 or 429 before that budget is spent. The score is won or lost on
// three things: sending every request to the best-rated free model that can serve
// it, never leaving a request unanswered when another provider can, and never
// rewriting a field the provider echoes back.
//
// How this entry does it:
//
//   * Routing reads the brief's ratings at runtime (readBrief) and ranks every
//     free model that meets the request's requirements by rating_milli. Nothing
//     about the practice trials' budgets, fail schedules or ratings is hardcoded:
//     the same code runs against the sealed trial's directory.
//   * Wire compatibility is part of the rank: a request that asked for JSON mode
//     on the OpenAI wire never goes to an Anthropic-wire provider, because that
//     wire cannot carry response_format and the echo would say json_mode false.
//   * Failover: on a 402, 429, 5xx or network error the request moves to the
//     next-best legal model; the same provider is never tried twice for one
//     request. Refusals are counted per provider across the trial, and a provider
//     with two consecutive refusals is skipped outright (it is likely exhausted
//     or out, and further calls there only burn the tie-break). A success resets
//     the count. Ties in rating prefer the provider whose wire matches the
//     client's (a passthrough cannot mistranslate), then the provider with the
//     fewest successes so far, which spreads the load across equal models.
//   * Every client-wire x provider-wire pair is translated in both directions
//     (lib/wire.ts covers the Anthropic client on an OpenAI provider; lib/translate.ts
//     covers the other four). The translations carry the echoed fields across
//     untouched: system text, the last user message's text, tool names in order,
//     max_tokens, stop sequences, image bytes, and JSON mode where the wire can
//     carry it. Streaming is translated event for event, and the OpenAI wire's
//     stream always carries the provider's usage in a final chunk (the entry asks
//     for it with stream_options.include_usage) while the Anthropic wire carries
//     it in message_start and message_delta and the Gemini wire in the final
//     frame's usageMetadata.
//   * GET /v1/models lists exactly the free model ids registered, in the shape of
//     the auth header: an x-api-key request gets Anthropic shape, an Authorization
//     request gets OpenAI shape. --list prints the catalog document byte-for-byte
//     identical to the register body.
//   * Everything is deterministic: no wall clock and no unseeded randomness. The
//     only state that carries across requests is the per-provider success/refusal
//     count, which is a pure function of what the providers answered.
//
// The server registers with the trial driver, answers the twenty requests one at
// a time, and polls /work until the trial is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsToOpenAI,
  anthropicToGemini,
  anthropicToOpenAIResponse,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIToAnthropicRequest,
  openAIToGemini,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take before it counts as refused, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider routing state across this trial: successes so far and consecutive refusals. */
interface ProviderState {
  successes: number;
  refusals: number;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { successes: 0, refusals: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

function noteRefusal(providerId: string): void {
  stateOf(providerId).refusals += 1;
}

function noteSuccess(providerId: string): void {
  const state = stateOf(providerId);
  state.refusals = 0;
  state.successes += 1;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const results = await Promise.all(
    brief.directory.providers.map(async (provider) => {
      const first = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      if (first.ok) return first;
      return probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    }),
  );
  probes = results;
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One free model that can serve the current request, with its rating and wire. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  wireMatch: boolean;
}

/**
 * Every free model that meets the request's requirements, in the order to try
 * them: highest rating first, then the provider whose wire matches the client's,
 * then the provider with the fewest successes so far, then provider and model id
 * so the order is deterministic. Providers with two consecutive refusals are
 * left out: they are likely exhausted or out, and more calls there only burn
 * the tie-break. A JSON-mode request never goes to an Anthropic-wire provider.
 */
function candidatesFor(requires: Requirement, needsJsonWire: boolean, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = stateOf(probe.provider.id);
    if (state.refusals >= 2) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (needsJsonWire && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, wireMatch: probe.provider.wire === clientWire });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (b.wireMatch ? 1 : 0) - (a.wireMatch ? 1 : 0) ||
      stateOf(a.probe.provider.id).successes - stateOf(b.probe.provider.id).successes ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream inference call: its URL, headers and translated body. */
interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

function upstreamCall(clientWire: 'openai' | 'anthropic', probe: Probe, modelId: string, body: Json, stream: boolean): UpstreamCall {
  const provider = probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';

  if (provider.wire === 'openai') {
    const translated = clientWire === 'openai' ? { ...body, model: modelId } : anthropicToOpenAI(body, modelId);
    if (stream) translated.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: translated };
  }
  if (provider.wire === 'anthropic') {
    const translated = clientWire === 'anthropic' ? { ...body, model: modelId } : openAIToAnthropicRequest(body, modelId);
    return { url: `${base}/v1/messages`, headers, body: translated };
  }
  // Gemini: the model travels in the URL path, not the body.
  const encoded = encodeURIComponent(modelId);
  const translated = clientWire === 'openai' ? openAIToGemini(body, modelId) : anthropicToGemini(body, modelId);
  return { url: `${base}/v1beta/models/${encoded}:${stream ? 'streamGenerateContent' : 'generateContent'}`, headers, body: translated };
}

/** A non-streamed upstream answer, translated into the client's wire. */
function translateResponse(clientWire: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', answer: Json, modelId: string): Json {
  if (clientWire === 'openai') {
    if (providerWire === 'openai') return answer;
    if (providerWire === 'anthropic') return anthropicToOpenAIResponse(answer, modelId);
    return geminiToOpenAIResponse(answer, modelId);
  }
  if (providerWire === 'openai') return openAIToAnthropic(answer, modelId);
  if (providerWire === 'anthropic') return answer;
  return geminiToAnthropicResponse(answer, modelId);
}

/**
 * An upstream event stream, translated into the client's wire as an async
 * generator of SSE frames. Only the cross-wire pairs reach here: when the
 * provider speaks the client's wire the bytes are piped untouched (see serve).
 */
async function* translateStream(
  clientWire: 'openai' | 'anthropic',
  providerWire: 'openai' | 'anthropic' | 'gemini',
  upstreamBody: ReadableStream<Uint8Array>,
  modelId: string,
): AsyncGenerator<string> {
  if (clientWire === 'anthropic' && providerWire === 'openai') {
    yield* anthropicEventsFromOpenAI(upstreamBody, modelId);
    return;
  }
  if (clientWire === 'openai' && providerWire === 'anthropic') {
    yield* anthropicEventsToOpenAI(upstreamBody, modelId);
    return;
  }
  if (clientWire === 'openai' && providerWire === 'gemini') {
    yield* geminiEventsToOpenAI(upstreamBody, modelId);
    return;
  }
  // Anthropic client on a Gemini provider.
  yield* geminiEventsToAnthropic(upstreamBody, modelId);
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Answer one chat or messages request, failing over through the candidates until
 * one answers. A refusal is never forwarded: the request moves to the next-best
 * legal model, and only when every candidate has refused does it get an error.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const needsJsonWire =
    clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const stream = body.stream === true;
  const candidates = candidatesFor(requires, needsJsonWire, clientWire);
  let lastError = 'no free model on the directory meets this request';

  for (const candidate of candidates) {
    const provider = candidate.probe.provider;
    const call = upstreamCall(clientWire, candidate.probe, candidate.model.id, body, stream);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (error) {
      noteRefusal(provider.id);
      lastError = error instanceof Error ? error.message : String(error);
      continue;
    }
    if (!upstream.ok) {
      noteRefusal(provider.id);
      lastError = `HTTP ${upstream.status}`;
      try {
        await upstream.text();
      } catch {
        // The error body is drained so the connection can close; a failure to read it is not one more refusal.
      }
      continue;
    }
    noteSuccess(provider.id);
    if (!stream) {
      const answer = (await upstream.json()) as Json;
      sendJson(res, 200, translateResponse(clientWire, provider.wire, answer, candidate.model.id));
      return;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end();
      return;
    }
    if (clientWire === provider.wire) {
      await pipeStream(upstream.body, res);
      return;
    }
    try {
      const events = translateStream(clientWire, provider.wire, upstream.body, candidate.model.id);
      for await (const event of events) res.write(event);
    } finally {
      res.end();
    }
    return;
  }

  if (clientWire === 'anthropic') sendError(res, 'anthropic', 503, 'overloaded_error', lastError);
  else sendError(res, 'openai', 503, 'no_free_model', lastError);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
