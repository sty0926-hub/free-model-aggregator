// The other four directions of the trial's wires.
//
// The starter's lib/wire.ts covers an Anthropic client on an OpenAI provider (and the
// OpenAI passthrough). This file covers the rest of the client-wire x provider-wire
// matrix, for a trial whose providers speak OpenAI, Anthropic and Gemini wires:
//
//   OpenAI client    -> Anthropic provider (zai): openAIToAnthropicRequest / anthropicEventsToOpenAI
//   OpenAI client    -> Gemini provider:    openAIToGemini / geminiEventsToOpenAI
//   Anthropic client -> Gemini provider:    anthropicToGemini / geminiEventsToAnthropic
//   (Anthropic client -> Anthropic provider is a passthrough with the model swapped;
//    OpenAI client -> OpenAI provider is the same.)
//
// Every translation is echo-preserving: the provider echoes the system text, the user
// text, the tool names, max_tokens, the stop sequences, the image bytes and the JSON
// mode it parsed, and the client compares that echo with what it sent. So each
// translator carries those fields across the wire boundary untouched, in the shape
// the receiving wire parses them from:
//
//   system text   OpenAI: system messages joined with "\n"
//                 Anthropic: the system string (or its text blocks joined with "\n")
//                 Gemini: systemInstruction parts, joined with "\n"
//   user text     OpenAI: the last user message's text blocks, joined with "\n"
//                 Anthropic: the last user message's text blocks, joined with "\n"
//                 Gemini: the last non-model content's text parts, joined with "\n"
//   image         OpenAI: image_url (a data: URL hashes its decoded bytes, a plain URL
//                 hashes the URL text); Anthropic: an image block with a base64 or url
//                 source (same hashes); Gemini: inlineData (base64 bytes) or fileData
//                 (a fileUri, hashed as a URL)
//   tools         the function names, in order, on every wire
//   max_tokens    OpenAI/Anthropic max_tokens; Gemini generationConfig.maxOutputTokens
//   stops         OpenAI stop; Anthropic stop_sequences; Gemini stopSequences
//   json mode     OpenAI response_format json_object/json_schema; Gemini
//                 generationConfig.responseMimeType "application/json"; the Anthropic
//                 wire cannot carry it at all, so a JSON-mode request is never routed
//                 to an Anthropic provider (main.ts enforces that).
//
// Stop reasons map by family: end_turn / stop / STOP are one family, max_tokens /
// length / MAX_TOKENS another, tool_use / tool_calls / function_call a third, and
// refusal / content_filter / SAFETY a fourth. The simulated providers always answer
// in the first family, so every client answer here reports "stop" on the OpenAI wire
// and "end_turn" on the Anthropic one.
//
// Usage is always the provider's own counts, in the client's wire's shape: the
// OpenAI wire reports prompt_tokens/completion_tokens, the Anthropic wire
// input_tokens/output_tokens, the Gemini wire usageMetadata.promptTokenCount and
// candidatesTokenCount. On a stream the OpenAI wire carries them in a final chunk
// with empty choices (the entry asks for it with stream_options.include_usage), the
// Anthropic wire in message_start (input) and message_delta (output), and the
// Gemini wire in the final frame's usageMetadata.
//
// Everything here is a pure function of its arguments: no clock and no randomness.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every listed model's `created`, as in lib/wire.ts. */
const FIXED_OPENAI_CREATED = 1767225600;

/** The text of OpenAI message content: a string, or its text parts joined with "\n". */
function openAIContentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of Anthropic message content: a string, or its text blocks joined with "\n". */
function anthropicContentText(content: unknown): string {
  return openAIContentText(content);
}

/** The text of every system message in an OpenAI messages array, joined with "\n". */
function openAISystemText(messages: unknown): string {
  if (!Array.isArray(messages)) return '';
  const parts: string[] = [];
  for (const m of messages) {
    if (!isRecord(m) || m.role !== 'system') continue;
    const text = openAIContentText(m.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = anthropicContentText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A stop list as an array: a string becomes one entry, an array passes through. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[3] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part: text or an inline image; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') {
      return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[3] } };
    }
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text or an inline image; null for the rest. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names
 * `model`: the system messages become the system string, image_url parts become
 * image blocks, tool definitions and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. response_format
 * is dropped: the Anthropic wire cannot carry it, and main.ts never routes a
 * JSON-mode request here.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const system = openAISystemText(source);
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
      else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length === 1 && blocks[0]?.type === 'text') messages.push({ role: 'assistant', content: (blocks[0] as Json).text });
      else if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    // A user message: text and image_url parts, in order.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length === 1 && blocks[0]?.type === 'text') messages.push({ role: 'user', content: (blocks[0] as Json).text });
    else if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. The model
 * travels in the URL path, not the body. response_format json_object/json_schema
 * becomes responseMimeType "application/json".
 */
export function openAIToGemini(body: Json, model: string): Json {
  void model;
  const source = Array.isArray(body.messages) ? body.messages : [];
  const system = openAISystemText(source);
  const contents: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string' && content !== '') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: toolResultText(raw.content) } } }],
      });
      continue;
    }
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = openAIPartToGemini(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  }
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The model
 * travels in the URL path, not the body. The Anthropic wire carries no JSON mode,
 * so no responseMimeType is ever set here.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  void model;
  const system = anthropicContentText(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string' && content !== '') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image blocks stay user parts; tool_result blocks
    // become a function content beside them.
    const content = raw.content;
    const parts: Json[] = [];
    const results: Json[] = [];
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          results.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
          continue;
        }
        const geminiPart = anthropicBlockToGemini(block);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (results.length > 0) contents.push({ role: 'function', parts: results });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini candidate: its parts' text, concatenated. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer's usage in {input, output} counts, or nulls when it carries none. */
function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiCandidateText(candidate);
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: geminiStopToOpenAI(candidate.finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages answer. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content: Json[] = [];
  const text = geminiCandidateText(candidate);
  if (text !== '') content.push({ type: 'text', text });
  const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  let toolIndex = 0;
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall)) continue;
    const call = part.functionCall;
    content.push({
      type: 'tool_use',
      id: `toolu_${toolIndex}`,
      name: typeof call.name === 'string' ? call.name : '',
      input: isRecord(call.args) ? call.args : {},
    });
    toolIndex += 1;
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = geminiUsageOf(body);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiStopToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One OpenAI chat chunk: the shape the client's stream parser reads. */
function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

/** One OpenAI usage chunk: empty choices, the provider's counts. */
function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/** The frames of a server-sent event stream, in order, as {event, data} pairs. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseSseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** One raw SSE frame's lines as {event, data}; data lines join with "\n". */
function parseSseFrame(frame: string): { event: string | null; data: string } {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/**
 * An Anthropic event stream as OpenAI chat chunks: text deltas become content
 * deltas, tool_use blocks become tool_calls deltas, and the stream closes with a
 * finish chunk, a usage chunk carrying the provider's counts, and [DONE].
 */
export async function* anthropicEventsToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  let finished = false;
  const toolIndexes = new Map<number, number>();
  let nextToolChunk = 0;

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield openAIChunk(model, {}, stopReason);
    yield openAIUsageChunk(model, inputTokens, outputTokens);
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          yield openAIChunk(model, { content: delta.text }, null);
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          let chunkIndex = toolIndexes.get(index);
          if (chunkIndex === undefined) {
            chunkIndex = nextToolChunk;
            nextToolChunk += 1;
            toolIndexes.set(index, chunkIndex);
          }
          yield openAIChunk(model, { tool_calls: [{ index: chunkIndex, function: { arguments: delta.partial_json } }] }, null);
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          let chunkIndex = toolIndexes.get(index);
          if (chunkIndex === undefined) {
            chunkIndex = nextToolChunk;
            nextToolChunk += 1;
            toolIndexes.set(index, chunkIndex);
          }
          yield openAIChunk(model, { tool_calls: [{ index: chunkIndex, id: typeof block.id === 'string' ? block.id : `toolu_${index}`, type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null);
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = anthropicStopToOpenAI(data.delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        break;
      }
      default:
        break;
    }
  }
  if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield* finish();
}

/**
 * A Gemini event stream as OpenAI chat chunks. Each frame carries one text piece;
 * the last carries the finishReason and the usageMetadata the usage chunk reports.
 */
export async function* geminiEventsToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  let finished = false;

  const finish = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield openAIChunk(model, {}, stopReason);
    yield openAIUsageChunk(model, inputTokens, outputTokens);
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const text = geminiCandidateText(candidate);
    if (text !== '') yield openAIChunk(model, { content: text }, null);
    if (typeof candidate.finishReason === 'string') stopReason = geminiStopToOpenAI(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      const usage = geminiUsageOf(data);
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
  }
  if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield* finish();
}

/**
 * A Gemini event stream as Anthropic events. The input count only arrives in the
 * final frame's usageMetadata, so the frames are read first and the event sequence
 * is emitted whole, with the provider's counts in message_start and message_delta.
 */
export async function* geminiEventsToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    text += geminiCandidateText(candidate);
    if (typeof candidate.finishReason === 'string') stopReason = geminiStopToAnthropic(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      const usage = geminiUsageOf(data);
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
  }
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  // The client's parser joins text deltas; one delta carries the whole text.
  if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
