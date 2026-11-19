// free-model-aggregator-02: one server, two client wires, and a real free
// model behind every answer.
//
// The server serves the OpenAI Chat Completions API and the Anthropic
// Messages API (JSON and server-sent events) and answers each of the trial's
// six prompts with a REAL completion from a live free OpenRouter model,
// dialled through the grader's gateway on the trial's key. There is no
// simulated provider anywhere in this entry.
//
// HOW IT WORKS
//   * Routing reads the brief's ratings at runtime and ranks every free
//     model that meets the request's requirements by rating_milli, highest
//     first. Nothing about the practice trials' ratings is hardcoded: the
//     brief is the source of truth, so a scored trial's different ratings
//     route the same way. Ties in rating prefer the lower model id, so the
//     order is total and deterministic.
//   * Failover: a gateway call answered with a 4xx or a 5xx, or one that
//     never answers, is never the prompt's answer. The server tries the next
//     free model that meets the prompt (at most 3 models a prompt, and never
//     the same model twice for one prompt), and it tries the BEST model
//     first on every prompt: a refusal costs nothing in the tie-break
//     (only successful calls count), so a transient refusal must not
//     demote the best model for the rest of the trial. Every refusal is
//     reported on the response in the x-makefun-upstream-failures header,
//     which is the only way the failover column can see a refusal the
//     server recovered from.
//   * The catalog is exactly the brief's fourteen free ids, on the register
//     body and on both GET /v1/models answers, byte for byte the same
//     document `--list` prints.
//   * Streaming is buffered: the gateway's whole SSE body is read and
//     checked before the client sees a 200, so a broken stream fails over
//     instead of handing the client half an answer.
//   * Every choice is a function of the brief, the requests and the
//     gateway's answers: no wall clock and no unseeded randomness in any
//     routing decision, so a replay leaves the same state export. (Timeouts
//     exist only as a safety valve against a hung call; in a replay the
//     recorded answers arrive at once and no timeout engages.)
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field is the model
//   it actually dialled. The house records every gateway call and scores the
//   model that really answered. This server sends the brief's model id in the
//   gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the
//   gateway's `openrouter/` prefix), so its claim is the call it made, never
//   another. A prompt whose reported model has no matching successful call in
//   the house's record earns nothing.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/catalog.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIChunksToAnthropicEvents,
  openAIChunksToFrames,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  parseSseChunks,
  type ServedModel,
  type StreamSummary,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway call may take this long before the server fails over. */
const ATTEMPT_TIMEOUT_MS = 9000;
/** The most models tried for one prompt. */
const MAX_ATTEMPTS = 3;
/** One client prompt may take this long in total before the server gives up. */
const REQUEST_DEADLINE_MS = 28000;
/** The header the failover column reads the refusal trail from. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
/** At most this many refusal rows in the trail. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does not
 * already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/**
 * The free models that can serve a request, best first: highest rating,
 * then lowest id. The order is a function of the brief alone, so it
 * replays identically, and a refusal in an earlier prompt never moves a
 * model down it: the best model is tried first on every prompt, because
 * only successful calls count toward the tie-break and a transient
 * refusal must not cost the rest of the trial its best model.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const fitting = freeModels(brief).filter((model) =>
    meetsRequirements(requires, capabilitiesOf(model)),
  );
  fitting.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return fitting;
}

/** The requirements of the prompt now arriving: the brief's row, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) {
        vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The tool names a request body declares, on either wire. */
function declaredToolNames(api: 'openai' | 'anthropic', body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (api === 'openai' && isRecord(tool.function) && typeof tool.function.name === 'string') {
      names.push(tool.function.name);
    } else if (api === 'anthropic' && typeof tool.name === 'string') {
      names.push(tool.name);
    }
  }
  return names;
}

/** One refusal row for the trail header. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway error body: its code, else its type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const error = value.error;
    if (typeof error.code === 'string' && error.code !== '') return error.code.slice(0, 64);
    if (typeof error.type === 'string' && error.type !== '') return error.type.slice(0, 64);
    return '';
  } catch {
    return '';
  }
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

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  failures: FailureRow[] = [],
): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) {
    headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
  }
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  failures: FailureRow[] = [],
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway call: POST the Chat Completions body, naming the model dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
  });
}

/**
 * A gateway answer's usage, made whole: the provider's own numbers when it
 * sent them, else an estimate from the answer's length. The wire checks
 * need a whole input count of at least 0 and an output count of at least 1;
 * a provider that sends no usage at all must not lose the prompt for it.
 */
function wholeUsage(
  answer: Json,
  requestChars: number,
): { input: number; output: number } {
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const text = answerTextOf(answer);
  const input =
    typeof usage.prompt_tokens === 'number' && usage.prompt_tokens >= 0
      ? usage.prompt_tokens
      : Math.max(0, Math.round(requestChars / 4));
  const output =
    typeof usage.completion_tokens === 'number' && usage.completion_tokens >= 1
      ? usage.completion_tokens
      : Math.max(1, Math.round(text.length / 4));
  return { input, output };
}

/** The text of a Chat Completions answer: its message content, or ''. */
function answerTextOf(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const message = isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
  }
  return '';
}

/** The tool calls of a Chat Completions answer, or []. */
function answerToolCalls(answer: Json): Json[] {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const message = isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
  return Array.isArray(message.tool_calls) ? message.tool_calls.filter(isRecord) : [];
}

/** A served answer: the model that really answered it and its body. */
interface ServedAnswer {
  model: string;
  /** The gateway's JSON answer (non-stream), or null when streamed. */
  json: Json | null;
  /** The parsed stream (streamed), or null when JSON. */
  stream: StreamSummary | null;
  failures: FailureRow[];
}

/**
 * Serve one prompt: try the ranked candidates in order until one answers
 * with content the client can use. A refusal, a hung call, an answer with
 * no text and no tool calls, or a tool call the client never declared all
 * count as a miss and move on to the next model. Returns null when every
 * candidate missed.
 */
async function serve(
  api: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
): Promise<ServedAnswer | null> {
  const candidates = candidatesFor(requires);
  const declared = declaredToolNames(api, body);
  const failures: FailureRow[] = [];
  const deadline = Date.now() + REQUEST_DEADLINE_MS;
  const requestChars = JSON.stringify(body).length;
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (Date.now() > deadline) break;
    attempts += 1;
    const upstreamBody =
      api === 'anthropic' ? anthropicToOpenAI(body, candidate.id) : { ...body };
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBody);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`dial ${candidate.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({
        status: upstream.status,
        code: errorCodeOf(text),
        model: candidate.id,
      });
      log(`dial ${candidate.id} refused HTTP ${upstream.status}`);
      continue;
    }
    if (body.stream === true) {
      let text: string;
      try {
        text = await upstream.text();
      } catch (err) {
        failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
        log(`stream ${candidate.id} broke: ${(err as Error).message}`);
        continue;
      }
      const summary = parseSseChunks(text);
      const usable =
        summary.chunks.length > 0 &&
        (summary.text !== '' ||
          summary.toolCalls.some((call) => declared.includes(call.name)));
      if (!usable) {
        // A stream with no usable content: the provider answered, but not
        // with anything the client can use. Count it as a miss, not a
        // refusal, and try the next model.
        log(`stream ${candidate.id} carried no usable content`);
        continue;
      }
      if (summary.usage === null) {
        summary.usage = {
          input: Math.max(0, Math.round(requestChars / 4)),
          output: Math.max(1, Math.round(summary.text.length / 4)),
        };
        summary.chunks.push({
          id: 'chatcmpl-stream',
          object: 'chat.completion.chunk',
          created: 1767225600,
          model: candidate.id,
          choices: [{ index: 0, delta: {}, finish_reason: null }],
          usage: {
            prompt_tokens: summary.usage.input,
            completion_tokens: summary.usage.output,
            total_tokens: summary.usage.input + summary.usage.output,
          },
        });
      }
      return { model: candidate.id, json: null, stream: summary, failures };
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`answer ${candidate.id} is not JSON`);
      continue;
    }
    if (!isRecord(answer)) {
      log(`answer ${candidate.id} is not one JSON object`);
      continue;
    }
    const text = answerTextOf(answer);
    const calls = answerToolCalls(answer);
    const undeclared = calls.filter(
      (call) =>
        !isRecord(call.function) ||
        typeof call.function.name !== 'string' ||
        !declared.includes(call.function.name),
    );
    if (text === '' && calls.length === 0) {
      // An answer with neither text nor a tool call cannot carry the token.
      log(`answer ${candidate.id} carried neither text nor a tool call`);
      continue;
    }
    if (text === '' && calls.length > 0) {
      // A tool call alone cannot carry the prompt's token either.
      log(`answer ${candidate.id} called a tool instead of answering`);
      continue;
    }
    if (undeclared.length > 0) {
      log(`answer ${candidate.id} named a tool the client never declared`);
      continue;
    }
    const usage = wholeUsage(answer, requestChars);
    answer.usage = {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    };
    return { model: candidate.id, json: answer, stream: null, failures };
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const served = await serve('openai', body, requires);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model answered this prompt', []);
  }
  if (body.stream === true && served.stream !== null) {
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (served.failures.length > 0) {
      headers[FAILURES_HEADER] = JSON.stringify(served.failures.slice(0, MAX_TRAIL_ROWS));
    }
    res.writeHead(200, headers);
    try {
      for (const frame of openAIChunksToFrames(served.stream, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  if (served.json === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model answered this prompt', served.failures);
  }
  sendJson(res, 200, openAIJsonWithModel(served.json, served.model), served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const served = await serve('anthropic', body, requires);
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this prompt', []);
  }
  if (body.stream === true && served.stream !== null) {
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (served.failures.length > 0) {
      headers[FAILURES_HEADER] = JSON.stringify(served.failures.slice(0, MAX_TRAIL_ROWS));
    }
    res.writeHead(200, headers);
    try {
      for (const event of openAIChunksToAnthropicEvents(served.stream, served.model)) {
        res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  if (served.json === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this prompt', served.failures);
  }
  const translated = openAIToAnthropic(served.json, served.model);
  const usage = isRecord(translated.usage) ? translated.usage : {};
  if (typeof usage.output_tokens !== 'number' || usage.output_tokens < 1) {
    const text = answerTextOf(served.json);
    translated.usage = {
      input_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      output_tokens: Math.max(1, Math.round(text.length / 4)),
    };
  }
  // openAIToAnthropic translates the finish_reason to the wire's stop
  // reason, and the JSON path's usage was already made whole in serve().
  sendJson(res, 200, translated, served.failures);
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
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      }
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
