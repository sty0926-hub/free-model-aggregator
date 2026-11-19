// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. The entry dials the grader's gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and relays the gateway's JSON or SSE answer, with every
//     `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this entry dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// Streaming is buffered, not relayed live: the entry reads the gateway's
// whole SSE body first, checks it carries content, and only then emits the
// client's events. These answers are a word or a JSON object, so the buffer
// is tiny, and buffering is what lets a broken stream fail over to the next
// model before the client has seen a 200.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './catalog.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) =>
      isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [],
    )
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or inline image. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: numberOrZero(usage.prompt_tokens),
      output_tokens: numberOrZero(usage.completion_tokens),
    },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One streamed tool call, accumulated from the stream's deltas. */
export interface StreamToolCall {
  id: string;
  name: string;
  /** The arguments JSON, concatenated from the deltas. */
  arguments: string;
}

/** What a buffered Chat Completions SSE body holds, once parsed. */
export interface StreamSummary {
  /** The text deltas joined, in order. */
  text: string;
  toolCalls: StreamToolCall[];
  /** The first non-null finish_reason seen, or null. */
  finishReason: string | null;
  /** The last usage object seen, or null. */
  usage: { input: number; output: number } | null;
  /** The chunk objects, in arrival order, for re-emission. */
  chunks: Json[];
  /** Whether the body ended with a [DONE] frame. */
  sawDone: boolean;
}

/**
 * A Chat Completions SSE body, parsed into its chunk objects. Frames are
 * split on the blank line that ends them; a frame cut across reads is held
 * until its end arrives. Lines that are not `data:` pass through ignored.
 */
export function parseSseChunks(bodyText: string): StreamSummary {
  const chunks: Json[] = [];
  let text = '';
  const toolByIndex = new Map<number, StreamToolCall>();
  let finishReason: string | null = null;
  let usage: { input: number; output: number } | null = null;
  let sawDone = false;
  const normalized = bodyText.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      chunks.push(chunk);
      if (isRecord(chunk.usage)) {
        usage = {
          input: numberOrZero(chunk.usage.prompt_tokens),
          output: numberOrZero(chunk.usage.completion_tokens),
        };
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (isRecord(choice.delta)) {
        const delta = choice.delta;
        if (typeof delta.content === 'string') text += delta.content;
        if (Array.isArray(delta.tool_calls)) {
          for (const call of delta.tool_calls) {
            if (!isRecord(call)) continue;
            const idx = typeof call.index === 'number' ? call.index : toolByIndex.size;
            let acc = toolByIndex.get(idx);
            if (acc === undefined) {
              acc = { id: '', name: '', arguments: '' };
              toolByIndex.set(idx, acc);
            }
            if (typeof call.id === 'string' && call.id !== '') acc.id = call.id;
            if (isRecord(call.function)) {
              if (typeof call.function.name === 'string' && call.function.name !== '') {
                acc.name = call.function.name;
              }
              if (typeof call.function.arguments === 'string') {
                acc.arguments += call.function.arguments;
              }
            }
          }
        }
      }
      if (typeof choice.finish_reason === 'string' && finishReason === null) {
        finishReason = choice.finish_reason;
      }
    }
  }
  const toolCalls = [...toolByIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, call]) => call);
  return { text, toolCalls, finishReason, usage, chunks, sawDone };
}

/**
 * The parsed chunks re-emitted as an OpenAI SSE stream, every chunk's `model`
 * the id the entry dialled, exactly one finish chunk, and [DONE] last. A
 * second finish_reason is stripped; a missing one is synthesized as `stop`.
 */
export function openAIChunksToFrames(summary: StreamSummary, model: string): string[] {
  const frames: string[] = [];
  let finished = false;
  for (const chunk of summary.chunks) {
    const copy: Json = { ...chunk, model };
    const choices = Array.isArray(copy.choices) ? copy.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && typeof choice.finish_reason === 'string') {
      if (finished) {
        choices[0] = { ...choice, finish_reason: null };
      } else {
        finished = true;
      }
    }
    frames.push(`data: ${JSON.stringify(copy)}\n\n`);
  }
  if (!finished) {
    frames.push(
      `data: ${JSON.stringify({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      })}\n\n`,
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames;
}

/**
 * The parsed chunks as the Anthropic event sequence: message_start on the
 * first event, content_block_start/delta/stop per text block or tool call,
 * then message_delta with the stop reason and usage, then message_stop.
 * `model` is the id the entry dialled.
 */
export function openAIChunksToAnthropicEvents(summary: StreamSummary, model: string): string[] {
  const events: string[] = [];
  const inputTokens = summary.usage !== null ? summary.usage.input : 0;
  const outputTokens = summary.usage !== null ? summary.usage.output : 0;
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  if (summary.text !== '') {
    const index = nextIndex;
    nextIndex += 1;
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    events.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: summary.text },
      }),
    );
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  for (const call of summary.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.arguments !== '') {
      events.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.arguments },
        }),
      );
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(summary.finishReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events;
}
