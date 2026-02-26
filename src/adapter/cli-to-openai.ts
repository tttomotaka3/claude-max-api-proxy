/**
 * Converts Claude CLI output to OpenAI-compatible response format
 */

import type { ClaudeCliAssistant, ClaudeCliResult } from "../types/claude-cli.js";
import type { OpenAIChatResponse, OpenAIChatChunk, OpenAIToolCall } from "../types/openai.js";

/**
 * Parsed result from extracting tool_use blocks from text
 */
export interface ParsedToolResponse {
  textContent: string;
  toolCalls: OpenAIToolCall[];
}

/**
 * Parse <tool_use> blocks from text and separate into text + tool_calls.
 * Returns tool calls in OpenAI format and the remaining text content.
 */
export function parseToolUseBlocks(text: string): ParsedToolResponse {
  const toolCalls: OpenAIToolCall[] = [];
  let callIndex = 0;

  // Match <tool_use>...</tool_use> blocks (possibly with whitespace)
  const toolUseRegex = /<tool_use>\s*([\s\S]*?)\s*<\/tool_use>/g;
  const cleanedText = text.replace(toolUseRegex, (_, jsonStr: string) => {
    try {
      const parsed = JSON.parse(jsonStr.trim());
      const name = parsed.name;
      const args = parsed.arguments || {};
      toolCalls.push({
        id: `call_${Date.now()}_${callIndex}`,
        type: "function",
        function: {
          name,
          arguments: JSON.stringify(args),
        },
      });
      callIndex++;
    } catch {
      // Failed to parse — leave in text as-is
      return `<tool_use>${jsonStr}</tool_use>`;
    }
    return ""; // Remove successfully parsed tool_use from text
  });

  return {
    textContent: cleanedText.trim(),
    toolCalls,
  };
}

/**
 * Extract text content from Claude CLI assistant message
 */
export function extractTextContent(message: ClaudeCliAssistant): string {
  return message.message.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("");
}

/**
 * Convert Claude CLI assistant message to OpenAI streaming chunk
 */
export function cliToOpenaiChunk(
  message: ClaudeCliAssistant,
  requestId: string,
  isFirst: boolean = false
): OpenAIChatChunk {
  const text = extractTextContent(message);

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(message.message.model),
    choices: [
      {
        index: 0,
        delta: {
          role: isFirst ? "assistant" : undefined,
          content: text,
        },
        finish_reason: message.message.stop_reason ? "stop" : null,
      },
    ],
  };
}

/**
 * Create a final "done" chunk for streaming
 */
export function createDoneChunk(requestId: string, model: string, finishReason: "stop" | "tool_calls" = "stop"): OpenAIChatChunk {
  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(model),
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: finishReason,
      },
    ],
  };
}

/**
 * Convert Claude CLI result to OpenAI non-streaming response.
 * When hasTools=true, parses tool_use blocks from the result text.
 */
export function cliResultToOpenai(
  result: ClaudeCliResult,
  requestId: string,
  hasTools: boolean = false
): OpenAIChatResponse {
  // Get model from modelUsage or default
  const modelName = result.modelUsage
    ? Object.keys(result.modelUsage)[0]
    : "claude-sonnet-4";

  let content: string | null = result.result;
  let toolCalls: OpenAIToolCall[] | undefined;
  let finishReason: "stop" | "tool_calls" = "stop";

  if (hasTools) {
    const parsed = parseToolUseBlocks(result.result);
    if (parsed.toolCalls.length > 0) {
      toolCalls = parsed.toolCalls;
      content = parsed.textContent || null;
      finishReason = "tool_calls";
    }
  }

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(modelName),
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content,
          ...(toolCalls ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: result.usage?.input_tokens || 0,
      completion_tokens: result.usage?.output_tokens || 0,
      total_tokens:
        (result.usage?.input_tokens || 0) + (result.usage?.output_tokens || 0),
    },
  };
}

/**
 * Normalize Claude model names to a consistent format
 * e.g., "claude-sonnet-4-5-20250929" -> "claude-sonnet-4"
 */
function normalizeModelName(model: string | undefined): string {
  if (!model) return "claude-sonnet-4";
  if (model.includes("opus")) return "claude-opus-4";
  if (model.includes("sonnet")) return "claude-sonnet-4";
  if (model.includes("haiku")) return "claude-haiku-4";
  return model;
}
