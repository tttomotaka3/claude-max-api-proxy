/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration
 */

import type { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs";
import * as path from "path";
import * as https from "https";
import { fileURLToPath } from "url";
import { ClaudeSubprocess } from "../subprocess/manager.js";
import { openaiToCli } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
  parseToolUseBlocks,
} from "../adapter/cli-to-openai.js";
import type { OpenAIChatRequest } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOG_FILE = path.resolve(__dirname, "../../logs/usage.jsonl");

const KNOWN_AGENTS = new Set(["tt", "az", "ym", "ar"]);

const RATE_LIMIT_PATTERNS = [
  /429/,
  /rate.?limit/i,
  /you'?ve hit your limit/i,
  /usage limit/i,
  /overloaded/i,
  /too many requests/i,
];

const AUTH_ERROR_PATTERNS = [
  /not logged in/i,
  /please run \/login/i,
  /run \/login/i,
];

const DISCORD_GENERAL_CHANNEL = "1475509846672674896";
const AUTH_ALERT_COOLDOWN_MS = 30 * 60 * 1000; // 30 min
let lastAuthAlertAt = 0;

function isRateLimitError(text: string): boolean {
  return RATE_LIMIT_PATTERNS.some((p) => p.test(text));
}

function isAuthError(text: string): boolean {
  return AUTH_ERROR_PATTERNS.some((p) => p.test(text));
}

function getDiscordBotToken(): string | null {
  if (process.env.DISCORD_BOT_TOKEN) {
    return process.env.DISCORD_BOT_TOKEN;
  }
  try {
    const configPath = path.resolve(
      process.env.HOME ?? "/Users/tt",
      ".openclaw/openclaw.json"
    );
    const config = JSON.parse(fs.readFileSync(configPath, "utf8")) as Record<string, unknown>;
    const accounts = (config?.channels as Record<string, unknown>)
      ?.discord as Record<string, unknown>;
    const ttAccount = (accounts?.accounts as Record<string, unknown>)
      ?.tt as Record<string, unknown>;
    return (ttAccount?.token as string) ?? null;
  } catch {
    return null;
  }
}

function notifyDiscordAuthError(message: string): void {
  const now = Date.now();
  if (now - lastAuthAlertAt < AUTH_ALERT_COOLDOWN_MS) {
    console.error("[AuthAlert] Cooldown active, skipping notification");
    return;
  }
  lastAuthAlertAt = now;

  const token = getDiscordBotToken();
  if (!token) {
    console.error("[AuthAlert] No Discord bot token available");
    return;
  }

  const content = `⚠️ **Claude CLI認証切れ検知**\nMac miniで \`claude\` を実行して \`/login\` してください。\n詳細: ${message.slice(0, 200)}`;
  const postData = JSON.stringify({ content });
  const options: https.RequestOptions = {
    hostname: "discord.com",
    port: 443,
    path: `/api/v10/channels/${DISCORD_GENERAL_CHANNEL}/messages`,
    method: "POST",
    headers: {
      "Authorization": `Bot ${token}`,
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(postData),
    },
  };

  const req = https.request(options, (res) => {
    res.on("data", () => {});
    res.on("end", () => {
      console.error(`[AuthAlert] Discord notification sent (status ${res.statusCode})`);
    });
  });
  req.on("error", (err) => {
    console.error("[AuthAlert] Failed to send Discord notification:", err.message);
  });
  req.write(postData);
  req.end();
}

function writeErrorLog(
  requestId: string,
  errorType: "rate_limit" | "auth_error" | "other",
  message: string,
  model: string,
  stream: boolean,
  sessionId?: string | null,
  agent?: string | null
): void {
  try {
    const logsDir = path.dirname(LOG_FILE);
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    const entry = {
      timestamp: new Date().toISOString(),
      requestId,
      sessionId: sessionId ?? null,
      agent: agent ?? null,
      model,
      error: errorType,
      errorMessage: message.slice(0, 500),
      stream,
    };
    fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + "\n", "utf8");
  } catch (err) {
    console.error("[writeErrorLog] Failed to write error log:", err);
  }
}

function extractAgent(req: Request): string | null {
  const agentHeader = req.headers["x-agent-id"] ?? req.headers["x-agent"];
  if (agentHeader && typeof agentHeader === "string") {
    return agentHeader;
  }
  const user = (req.body as OpenAIChatRequest)?.user;
  if (user && typeof user === "string" && KNOWN_AGENTS.has(user.toLowerCase())) {
    return user.toLowerCase();
  }
  return null;
}

function writeUsageLog(
  requestId: string,
  result: ClaudeCliResult,
  model: string,
  stream: boolean,
  hasToolCalls: boolean,
  sessionId?: string | null,
  agent?: string | null
): void {
  try {
    const logsDir = path.dirname(LOG_FILE);
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    const entry = {
      timestamp: new Date().toISOString(),
      requestId,
      sessionId: sessionId ?? null,
      agent: agent ?? null,
      model,
      inputTokens: result.usage.input_tokens,
      outputTokens: result.usage.output_tokens,
      cacheCreationTokens: result.usage.cache_creation_input_tokens ?? 0,
      cacheReadTokens: result.usage.cache_read_input_tokens ?? 0,
      totalCostUsd: result.total_cost_usd,
      durationMs: result.duration_ms,
      durationApiMs: result.duration_api_ms,
      numTurns: result.num_turns,
      modelUsage: result.modelUsage,
      hasToolCalls,
      stream,
    };
    fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + "\n", "utf8");
  } catch (err) {
    console.error("[writeUsageLog] Failed to write usage log:", err);
  }
}

/**
 * Handle POST /v1/chat/completions
 *
 * Main endpoint for chat requests, supports both streaming and non-streaming
 */
export async function handleChatCompletions(
  req: Request,
  res: Response
): Promise<void> {
  const requestId = uuidv4().replace(/-/g, "").slice(0, 24);
  const body = req.body as OpenAIChatRequest;
  const stream = body.stream === true;

  try {
    // Validate request
    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array",
          type: "invalid_request_error",
          code: "invalid_messages",
        },
      });
      return;
    }

    // Convert to CLI input format
    const cliInput = openaiToCli(body);
    console.error(`[DEBUG] hasTools=${cliInput.hasTools}, toolCount=${body.tools?.length || 0}, model=${body.model}`);
    if (cliInput.hasTools) {
      console.error(`[DEBUG] Tool names: ${body.tools!.map(t => t.function.name).join(', ')}`);
    }
    const subprocess = new ClaudeSubprocess();

    const agent = extractAgent(req);
    if (stream) {
      await handleStreamingResponse(req, res, subprocess, cliInput, requestId, agent);
    } else {
      await handleNonStreamingResponse(res, subprocess, cliInput, requestId, agent);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message,
          type: "server_error",
          code: null,
        },
      });
    }
  }
}

/**
 * Handle streaming response (SSE)
 *
 * When tools are present, we buffer the full response text to detect <tool_use>
 * blocks at the end. Text before tool_use blocks is streamed normally.
 * When tool_use blocks are found, they are sent as tool_calls chunks.
 */
async function handleStreamingResponse(
  req: Request,
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  agent: string | null
): Promise<void> {
  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);

  // CRITICAL: Flush headers immediately to establish SSE connection
  res.flushHeaders();

  return new Promise<void>((resolve, reject) => {
    let isFirst = true;
    let lastModel = "claude-sonnet-4";
    let isComplete = false;
    const hasTools = cliInput.hasTools;

    // When tools are present, buffer all text to parse tool_use at the end
    let fullTextBuffer = "";

    // Handle actual client disconnect (response stream closed)
    res.on("close", () => {
      if (!isComplete) {
        subprocess.kill();
      }
      resolve();
    });

    // Handle streaming content deltas
    subprocess.on("content_delta", (event: ClaudeCliStreamEvent) => {
      const text = event.event.delta?.text || "";
      if (text && !res.writableEnded) {
        if (hasTools) {
          // Buffer text — we'll flush it on result
          fullTextBuffer += text;
        } else {
          // No tools — stream directly
          const chunk = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: lastModel,
            choices: [{
              index: 0,
              delta: {
                role: isFirst ? "assistant" : undefined,
                content: text,
              },
              finish_reason: null,
            }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          isFirst = false;
        }
      }
    });

    // Handle final assistant message (for model name)
    subprocess.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      if (res.writableEnded) {
        resolve();
        return;
      }

      // Check result text for auth errors (auth errors come through result.result, not stderr)
      if (result.result && isAuthError(result.result)) {
        console.error("[Streaming] Auth error detected in result:", result.result.slice(0, 200));
        writeErrorLog(requestId, "auth_error", result.result, lastModel, true, cliInput.sessionId, agent);
        notifyDiscordAuthError(result.result);
      }

      if (hasTools) {
        // Use the result text (more complete than streamed buffer)
        const resultText = result.result || fullTextBuffer;
        console.error(`[DEBUG] Result text (last 500 chars): ${resultText.slice(-500)}`);
        console.error(`[DEBUG] Contains <tool_use>: ${resultText.includes('<tool_use>')}`);
        const parsed = parseToolUseBlocks(resultText);
        console.error(`[DEBUG] Parsed toolCalls: ${parsed.toolCalls.length}, textContent length: ${parsed.textContent.length}`);
        writeUsageLog(requestId, result, lastModel, true, parsed.toolCalls.length > 0, cliInput.sessionId, agent);

        // Send text content if any
        if (parsed.textContent) {
          const textChunk = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: lastModel,
            choices: [{
              index: 0,
              delta: {
                role: "assistant" as const,
                content: parsed.textContent,
              },
              finish_reason: null,
            }],
          };
          res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
        }

        if (parsed.toolCalls.length > 0) {
          // Send tool_calls chunks
          for (let i = 0; i < parsed.toolCalls.length; i++) {
            const tc = parsed.toolCalls[i];
            const toolChunk = {
              id: `chatcmpl-${requestId}`,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: lastModel,
              choices: [{
                index: 0,
                delta: {
                  tool_calls: [{
                    index: i,
                    id: tc.id,
                    type: "function" as const,
                    function: {
                      name: tc.function.name,
                      arguments: tc.function.arguments,
                    },
                  }],
                },
                finish_reason: null,
              }],
            };
            res.write(`data: ${JSON.stringify(toolChunk)}\n\n`);
          }

          // Final chunk with finish_reason=tool_calls
          const doneChunk = createDoneChunk(requestId, lastModel, "tool_calls");
          res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
        } else {
          // No tool calls found — normal finish
          const doneChunk = createDoneChunk(requestId, lastModel);
          res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
        }
      } else {
        // No tools — normal finish
        writeUsageLog(requestId, result, lastModel, true, false, cliInput.sessionId, agent);
        const doneChunk = createDoneChunk(requestId, lastModel);
        res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
      }

      res.write("data: [DONE]\n\n");
      res.end();
      resolve();
    });

    subprocess.on("stderr", (text: string) => {
      if (isAuthError(text)) {
        console.error("[Streaming] Auth error detected in stderr:", text.slice(0, 200));
        writeErrorLog(requestId, "auth_error", text, lastModel, true, cliInput.sessionId, agent);
        notifyDiscordAuthError(text);
      } else if (isRateLimitError(text)) {
        console.error("[Streaming] Rate limit detected in stderr:", text.slice(0, 200));
        writeErrorLog(requestId, "rate_limit", text, lastModel, true, cliInput.sessionId, agent);
      }
    });

    subprocess.on("error", (error: Error) => {
      console.error("[Streaming] Error:", error.message);
      const errorType = isAuthError(error.message) ? "auth_error" : isRateLimitError(error.message) ? "rate_limit" : "other";
      if (errorType === "auth_error") notifyDiscordAuthError(error.message);
      writeErrorLog(requestId, errorType, error.message, lastModel, true, cliInput.sessionId, agent);
      if (!res.writableEnded) {
        res.write(
          `data: ${JSON.stringify({
            error: { message: error.message, type: "server_error", code: null },
          })}\n\n`
        );
        res.end();
      }
      resolve();
    });

    subprocess.on("close", (code: number | null) => {
      // Subprocess exited - ensure response is closed
      if (!res.writableEnded) {
        if (code !== 0 && !isComplete) {
          res.write(`data: ${JSON.stringify({
            error: { message: `Process exited with code ${code}`, type: "server_error", code: null },
          })}\n\n`);
        }
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    });

    // Start the subprocess
    subprocess.start(cliInput.prompt, {
      model: cliInput.model,
      sessionId: cliInput.sessionId,
    }).catch((err) => {
      console.error("[Streaming] Subprocess start error:", err);
      reject(err);
    });
  });
}

/**
 * Handle non-streaming response
 */
async function handleNonStreamingResponse(
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  agent: string | null
): Promise<void> {
  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;

    subprocess.on("result", (result: ClaudeCliResult) => {
      finalResult = result;
      // Check result text for auth errors
      if (result.result && isAuthError(result.result)) {
        console.error("[NonStreaming] Auth error detected in result:", result.result.slice(0, 200));
        writeErrorLog(requestId, "auth_error", result.result, "unknown", false, cliInput.sessionId, agent);
        notifyDiscordAuthError(result.result);
      }
    });

    subprocess.on("stderr", (text: string) => {
      if (isAuthError(text)) {
        console.error("[NonStreaming] Auth error detected in stderr:", text.slice(0, 200));
        writeErrorLog(requestId, "auth_error", text, "unknown", false, cliInput.sessionId, agent);
        notifyDiscordAuthError(text);
      } else if (isRateLimitError(text)) {
        console.error("[NonStreaming] Rate limit detected in stderr:", text.slice(0, 200));
        writeErrorLog(requestId, "rate_limit", text, "unknown", false, cliInput.sessionId, agent);
      }
    });

    subprocess.on("error", (error: Error) => {
      console.error("[NonStreaming] Error:", error.message);
      const errorType = isAuthError(error.message) ? "auth_error" : isRateLimitError(error.message) ? "rate_limit" : "other";
      if (errorType === "auth_error") notifyDiscordAuthError(error.message);
      writeErrorLog(requestId, errorType, error.message, "unknown", false, cliInput.sessionId, agent);
      res.status(500).json({
        error: {
          message: error.message,
          type: "server_error",
          code: null,
        },
      });
      resolve();
    });

    subprocess.on("close", (code: number | null) => {
      if (finalResult) {
        writeUsageLog(requestId, finalResult, finalResult.modelUsage ? Object.keys(finalResult.modelUsage)[0] ?? "unknown" : "unknown", false, cliInput.hasTools);
        res.json(cliResultToOpenai(finalResult, requestId, cliInput.hasTools));
      } else if (!res.headersSent) {
        res.status(500).json({
          error: {
            message: `Claude CLI exited with code ${code} without response`,
            type: "server_error",
            code: null,
          },
        });
      }
      resolve();
    });

    // Start the subprocess
    subprocess
      .start(cliInput.prompt, {
        model: cliInput.model,
        sessionId: cliInput.sessionId,
      })
      .catch((error) => {
        res.status(500).json({
          error: {
            message: error.message,
            type: "server_error",
            code: null,
          },
        });
        resolve();
      });
  });
}

/**
 * Handle GET /v1/usage
 *
 * Aggregates usage.jsonl by date, model, agent, pj, and hour
 * Supports query parameters: from, to, agent, model
 */
export function handleUsage(req: Request, res: Response): void {
  const { from, to, agent: agentFilter, model: modelFilter } = req.query as Record<string, string | undefined>;

  interface AggBucket {
    requests: number;
    inputTokens: number;
    outputTokens: number;
    costUsd: number;
  }

  const summary = {
    totalRequests: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalCostUsd: 0,
    byDate: {} as Record<string, AggBucket>,
    byModel: {} as Record<string, AggBucket>,
    byAgent: {} as Record<string, AggBucket>,
    byPj: {} as Record<string, AggBucket>,
    byHour: {} as Record<string, AggBucket>,
  };

  function addTo(bucket: Record<string, AggBucket>, key: string, tokens: { input: number; output: number; cost: number }) {
    if (!bucket[key]) {
      bucket[key] = { requests: 0, inputTokens: 0, outputTokens: 0, costUsd: 0 };
    }
    bucket[key].requests++;
    bucket[key].inputTokens += tokens.input;
    bucket[key].outputTokens += tokens.output;
    bucket[key].costUsd += tokens.cost;
  }

  try {
    if (!fs.existsSync(LOG_FILE)) {
      res.json({ summary });
      return;
    }

    const lines = fs.readFileSync(LOG_FILE, "utf8").split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      let entry: Record<string, unknown>;
      try {
        entry = JSON.parse(line);
      } catch {
        continue;
      }

      const ts = String(entry.timestamp ?? "");
      const dateStr = ts.slice(0, 10); // YYYY-MM-DD

      // Date filter
      if (from && dateStr < from) continue;
      if (to && dateStr > to) continue;

      const entryAgent = String(entry.agent ?? "unknown");
      const entryModel = String(entry.model ?? "unknown");
      const entryPj = entry.pj ? String(entry.pj) : null;
      const hourStr = ts.slice(0, 13); // YYYY-MM-DDTHH

      // Field filters
      if (agentFilter && entryAgent !== agentFilter) continue;
      if (modelFilter && entryModel !== modelFilter) continue;

      const inputTokens = Number(entry.inputTokens ?? 0);
      const outputTokens = Number(entry.outputTokens ?? 0);
      const costUsd = Number(entry.totalCostUsd ?? 0);
      const tokens = { input: inputTokens, output: outputTokens, cost: costUsd };

      summary.totalRequests++;
      summary.totalInputTokens += inputTokens;
      summary.totalOutputTokens += outputTokens;
      summary.totalCostUsd += costUsd;

      addTo(summary.byDate, dateStr, tokens);
      addTo(summary.byModel, entryModel, tokens);
      addTo(summary.byAgent, entryAgent, tokens);
      if (entryPj) addTo(summary.byPj, entryPj, tokens);
      if (hourStr) addTo(summary.byHour, hourStr, tokens);
    }

    // Round cost to 8 decimal places
    summary.totalCostUsd = Math.round(summary.totalCostUsd * 1e8) / 1e8;
    for (const bucket of [summary.byDate, summary.byModel, summary.byAgent, summary.byPj, summary.byHour]) {
      for (const key of Object.keys(bucket)) {
        bucket[key].costUsd = Math.round(bucket[key].costUsd * 1e8) / 1e8;
      }
    }

    res.json({ summary });
  } catch (err) {
    console.error("[handleUsage] Error:", err);
    res.status(500).json({
      error: { message: "Failed to read usage log", type: "server_error", code: null },
    });
  }
}

/**
 * Handle GET /v1/models
 *
 * Returns available models
 */
export function handleModels(_req: Request, res: Response): void {
  res.json({
    object: "list",
    data: [
      {
        id: "claude-opus-4",
        object: "model",
        owned_by: "anthropic",
        created: Math.floor(Date.now() / 1000),
      },
      {
        id: "claude-sonnet-4",
        object: "model",
        owned_by: "anthropic",
        created: Math.floor(Date.now() / 1000),
      },
      {
        id: "claude-haiku-4",
        object: "model",
        owned_by: "anthropic",
        created: Math.floor(Date.now() / 1000),
      },
    ],
  });
}

/**
 * Handle GET /health
 *
 * Health check endpoint
 */
export function handleHealth(_req: Request, res: Response): void {
  res.json({
    status: "ok",
    provider: "claude-code-cli",
    timestamp: new Date().toISOString(),
  });
}
