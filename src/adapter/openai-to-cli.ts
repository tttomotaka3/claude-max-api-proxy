/**
 * Converts OpenAI chat request format to Claude CLI input
 */

import type { OpenAIChatRequest, OpenAITool, OpenAIChatMessage } from "../types/openai.js";

export type ClaudeModel = "opus" | "sonnet" | "haiku";

export interface CliInput {
  prompt: string;
  model: ClaudeModel;
  sessionId?: string;
  hasTools: boolean;
}

const MODEL_MAP: Record<string, ClaudeModel> = {
  // Direct model names
  "claude-opus-4": "opus",
  "claude-sonnet-4": "sonnet",
  "claude-haiku-4": "haiku",
  // With provider prefix
  "claude-code-cli/claude-opus-4": "opus",
  "claude-code-cli/claude-sonnet-4": "sonnet",
  "claude-code-cli/claude-haiku-4": "haiku",
  // Aliases
  "opus": "opus",
  "sonnet": "sonnet",
  "haiku": "haiku",
};

/**
 * Extract Claude model alias from request model string
 */
export function extractModel(model: string): ClaudeModel {
  // Try direct lookup
  if (MODEL_MAP[model]) {
    return MODEL_MAP[model];
  }

  // Try stripping provider prefix
  const stripped = model.replace(/^claude-code-cli\//, "");
  if (MODEL_MAP[stripped]) {
    return MODEL_MAP[stripped];
  }

  // Default to opus (Claude Max subscription)
  return "opus";
}

function extractContent(content: string | Array<{ type: string; text?: string; [key: string]: unknown }> | null): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  return content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("");
}

/**
 * Convert OpenAI tools array to a text instruction block for the prompt.
 * Claude will output <tool_use> blocks when it wants to call a tool.
 */
function toolsToPromptBlock(tools: OpenAITool[]): string {
  const lines: string[] = [
    "<available_tools>",
    "You have access to the following tools. When you need to call a tool, output EXACTLY this format (no markdown fences around it):",
    "",
    "<tool_use>",
    '{"name": "tool_name", "arguments": {"param": "value"}}',
    "</tool_use>",
    "",
    "You may output text before or after a tool call. You may make multiple tool calls in a single response.",
    "IMPORTANT: The JSON inside <tool_use> must be valid JSON on a single line.",
    "",
    "Available tools:",
    "",
  ];

  for (const tool of tools) {
    const fn = tool.function;
    lines.push(`## ${fn.name}`);
    if (fn.description) {
      lines.push(fn.description);
    }
    if (fn.parameters) {
      lines.push(`Parameters: ${JSON.stringify(fn.parameters)}`);
    }
    lines.push("");
  }

  lines.push("</available_tools>");
  return lines.join("\n");
}

/**
 * Convert messages array (including tool messages) to a single prompt string.
 */
function messagesToPrompt(messages: OpenAIChatMessage[], tools?: OpenAITool[]): string {
  const parts: string[] = [];

  // Inject tools as part of the first system message or as a standalone block
  let toolsInjected = false;

  for (const msg of messages) {
    switch (msg.role) {
      case "system": {
        let text = extractContent(msg.content);
        // Inject tools into the first system message
        if (!toolsInjected && tools && tools.length > 0) {
          text += "\n\n" + toolsToPromptBlock(tools);
          toolsInjected = true;
        }
        parts.push(`<system>\n${text}\n</system>\n`);
        break;
      }

      case "user":
        parts.push(extractContent(msg.content));
        break;

      case "assistant": {
        // Reconstruct assistant message including any tool_calls
        const textContent = extractContent(msg.content);
        const toolCalls = msg.tool_calls;
        if (toolCalls && toolCalls.length > 0) {
          const callBlocks = toolCalls.map((tc) =>
            `<tool_use>\n${JSON.stringify({ name: tc.function.name, arguments: JSON.parse(tc.function.arguments) })}\n</tool_use>`
          ).join("\n");
          const combined = [textContent, callBlocks].filter(Boolean).join("\n\n");
          parts.push(`<previous_response>\n${combined}\n</previous_response>\n`);
        } else {
          parts.push(`<previous_response>\n${textContent}\n</previous_response>\n`);
        }
        break;
      }

      case "tool": {
        // Tool result message — format as tool_result block
        const toolCallId = msg.tool_call_id || "unknown";
        const result = extractContent(msg.content);
        parts.push(`<tool_result tool_call_id="${toolCallId}">\n${result}\n</tool_result>\n`);
        break;
      }
    }
  }

  // If tools exist but no system message was found, prepend tools block
  if (!toolsInjected && tools && tools.length > 0) {
    parts.unshift(`<system>\n${toolsToPromptBlock(tools)}\n</system>\n`);
  }

  return parts.join("\n").trim();
}

/**
 * Convert OpenAI chat request to CLI input format
 */
export function openaiToCli(request: OpenAIChatRequest): CliInput {
  const hasTools = !!(request.tools && request.tools.length > 0);
  return {
    prompt: messagesToPrompt(request.messages, request.tools),
    model: extractModel(request.model),
    sessionId: request.user,
    hasTools,
  };
}
