/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration
 */

import type { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { ClaudeSubprocess } from "../subprocess/manager.js";
import { openaiToCli } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
  parseToolUseBlocks,
} from "../adapter/cli-to-openai.js";
import type { OpenAIChatRequest } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

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

    if (stream) {
      await handleStreamingResponse(req, res, subprocess, cliInput, requestId);
    } else {
      await handleNonStreamingResponse(res, subprocess, cliInput, requestId);
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
  requestId: string
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

      if (hasTools) {
        // Use the result text (more complete than streamed buffer)
        const resultText = result.result || fullTextBuffer;
        console.error(`[DEBUG] Result text (last 500 chars): ${resultText.slice(-500)}`);
        console.error(`[DEBUG] Contains <tool_use>: ${resultText.includes('<tool_use>')}`);
        const parsed = parseToolUseBlocks(resultText);
        console.error(`[DEBUG] Parsed toolCalls: ${parsed.toolCalls.length}, textContent length: ${parsed.textContent.length}`);

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
        const doneChunk = createDoneChunk(requestId, lastModel);
        res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
      }

      res.write("data: [DONE]\n\n");
      res.end();
      resolve();
    });

    subprocess.on("error", (error: Error) => {
      console.error("[Streaming] Error:", error.message);
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
  requestId: string
): Promise<void> {
  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;

    subprocess.on("result", (result: ClaudeCliResult) => {
      finalResult = result;
    });

    subprocess.on("error", (error: Error) => {
      console.error("[NonStreaming] Error:", error.message);
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
