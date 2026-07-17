import json
import os
import signal
import socket
import time
import uuid
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

UPSTREAM_API_URL = os.getenv("UPSTREAM_API_URL", "http://ai-api.zy-robot.cn/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

if not UPSTREAM_API_KEY:
    raise RuntimeError("未设置环境变量 UPSTREAM_API_KEY")

app = FastAPI(title="Local OpenAI Gateway")


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _find_gateway_pid_by_port(port: int) -> int | None:
    proc_root = "/proc"
    target_script = os.path.abspath(__file__)
    port_text = str(port)

    for pid in os.listdir(proc_root):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue

        cmdline_path = os.path.join(proc_root, pid, "cmdline")
        try:
            with open(cmdline_path, "rb") as file:
                raw = file.read()
        except OSError:
            continue

        if not raw:
            continue

        cmdline = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\x00") if part]
        if target_script not in cmdline:
            continue

        for part in cmdline:
            if port_text in part or target_script == part:
                return int(pid)

    return None


def _cleanup_previous_gateway_process() -> None:
    if not _is_port_in_use(HOST, PORT):
        return

    gateway_pid = _find_gateway_pid_by_port(PORT)
    if gateway_pid is None:
        raise RuntimeError(f"端口 {PORT} 已被其他进程占用，未自动清理")

    # 仅当占用端口的是当前网关旧进程时，才自动结束，避免误杀其他服务
    os.kill(gateway_pid, signal.SIGTERM)

    for _ in range(20):
        if not _is_port_in_use(HOST, PORT):
            return
        time.sleep(0.2)

    raise RuntimeError(f"旧网关进程 {gateway_pid} 未能及时释放端口 {PORT}")


def _make_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json",
    }


def _normalize_message_content(content: Any, role: str = "user") -> list[dict[str, Any]]:
    # 根据消息角色区分 input_text 与 output_text，避免把历史 assistant 输出误当成用户输入
    text_type = "output_text" if role == "assistant" else "input_text"

    if isinstance(content, str):
        return [{"type": text_type, "text": content}]

    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": text_type, "text": item})
                continue

            if not isinstance(item, dict):
                normalized.append({"type": text_type, "text": str(item)})
                continue

            item_type = item.get("type")
            if item_type == "text":
                normalized.append({"type": text_type, "text": item.get("text", "")})
            elif item_type in {"input_text", "output_text"}:
                normalized.append({"type": text_type, "text": item.get("text", "")})
            elif item_type in {"input_image", "image_url"}:
                normalized.append(item)
            else:
                normalized.append({"type": text_type, "text": json.dumps(item, ensure_ascii=False)})
        return normalized

    if content is None:
        return []

    return [{"type": text_type, "text": str(content)}]


def _stringify_content(content: Any) -> str:
    # tool 输出需要尽量保留可读文本，供 Responses API 的 function_call_output 使用
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue

            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
            elif item_type == "image_url":
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "".join(parts)

    if content is None:
        return ""

    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _normalize_call_id(call_id: Any) -> str:
    call_id_text = str(call_id or "")
    if not call_id_text:
        return f"fc_{uuid.uuid4().hex}"
    if call_id_text.startswith("fc_"):
        return call_id_text
    if call_id_text.startswith("call_"):
        return f"fc_{call_id_text[5:]}"
    return f"fc_{call_id_text}"


def _convert_assistant_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    converted: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue

        normalized_call_id = _normalize_call_id(tool_call.get("id"))
        converted.append(
            {
                "type": "function_call",
                "id": normalized_call_id,
                "call_id": normalized_call_id,
                "name": function.get("name"),
                "arguments": function.get("arguments", ""),
            }
        )

    return converted


def _chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 将 Chat Completions 的消息历史拆成 Responses API 可识别的普通消息、函数调用和函数输出
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = message.get("content")

        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": _normalize_call_id(message.get("tool_call_id")),
                    "output": _stringify_content(content),
                }
            )
            continue

        normalized_content = _normalize_message_content(content, role)
        if normalized_content:
            result.append({"type": "message", "role": role, "content": normalized_content})

        # assistant 的 tool_calls 不能当普通文本透传，需要展开成函数调用项
        if role == "assistant":
            result.extend(_convert_assistant_tool_calls(message.get("tool_calls")))
    return result


def _convert_chat_tools_to_responses_tools(tools: Any) -> list[dict[str, Any]]:
    # Chat Completions 的 function tools 需要展开成 Responses API 的函数工具定义
    if not isinstance(tools, list):
        return []

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            converted.append(tool)
            continue

        function = tool.get("function", {})
        if not isinstance(function, dict):
            continue

        converted.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )

    return converted


def _convert_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function", {})
        if isinstance(function, dict):
            return {"type": "function", "name": function.get("name")}
    return tool_choice


def _build_upstream_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": body.get("model"),
        "input": _chat_messages_to_responses_input(body.get("messages", [])),
        "stream": body.get("stream", False),
    }

    passthrough_fields = [
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "metadata",
        "instructions",
        "store",
        "user",
        "presence_penalty",
        "frequency_penalty",
    ]
    for field in passthrough_fields:
        if field in body:
            payload[field] = body[field]

    if "max_tokens" in payload and "max_output_tokens" not in payload:
        payload["max_output_tokens"] = payload.pop("max_tokens")

    converted_tools = _convert_chat_tools_to_responses_tools(body.get("tools"))
    if converted_tools:
        payload["tools"] = converted_tools

    if "tool_choice" in body:
        payload["tool_choice"] = _convert_tool_choice(body.get("tool_choice"))

    return payload


def _extract_text_from_output(response_json: dict[str, Any]) -> str:
    # 从 Responses API 响应中尽量提取最终文本内容
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    texts: list[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text_value = content.get("text")
            if isinstance(text_value, str):
                texts.append(text_value)
    return "".join(texts)


def _responses_to_chat_completion(response_json: dict[str, Any], model: str | None) -> dict[str, Any]:
    # 将非流式 Responses API JSON 转换为 Chat Completions JSON
    text = _extract_text_from_output(response_json)
    response_id = response_json.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    created = int(response_json.get("created_at") or time.time())

    usage = response_json.get("usage", {}) if isinstance(response_json.get("usage"), dict) else {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": response_json.get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": response_json.get("status", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _error_response_from_upstream(status_code: int, body_text: str, content_type: str | None) -> Response:
    media_type = content_type or "application/json"
    return Response(content=body_text, status_code=status_code, media_type=media_type)


def _build_chat_stream_chunk(response_id: str, model: str | None, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _stream_upstream_as_chat_completions(payload: dict[str, Any], model: str | None) -> AsyncGenerator[dict[str, str], None]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 先发送一个 role chunk，兼容常见 Chat Completions SSE 消费端
    yield {"data": json.dumps(_build_chat_stream_chunk(response_id, model, {"role": "assistant"}), ensure_ascii=False)}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{UPSTREAM_API_URL.rstrip('/')}/responses",
                headers=_make_headers(),
                json=payload,
            ) as upstream_response:
                if upstream_response.status_code >= 400:
                    error_body = await upstream_response.aread()
                    raise httpx.HTTPStatusError(
                        "Upstream returned error status",
                        request=upstream_response.request,
                        response=upstream_response,
                    ) from RuntimeError(error_body.decode("utf-8", errors="ignore"))

                async for line in upstream_response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    # 将 Responses API 的文本增量事件转换为 Chat Completions 的 delta.content
                    delta_text = None
                    if event_type in {"response.output_text.delta", "output_text.delta"}:
                        delta_text = event.get("delta", "")
                    elif event_type in {"response.completed", "response.output_text.done", "output_text.done"}:
                        delta_text = None
                    elif isinstance(event.get("delta"), str):
                        delta_text = event.get("delta")

                    if delta_text:
                        chunk = _build_chat_stream_chunk(response_id, model, {"content": delta_text})
                        yield {"data": json.dumps(chunk, ensure_ascii=False)}

    except httpx.HTTPStatusError as exc:
        response = exc.response
        body_text = ""
        if exc.__cause__ is not None:
            body_text = str(exc.__cause__)
        elif response is not None:
            body_text = response.text
        error_chunk = {
            "error": {
                "message": body_text or "上游服务返回错误",
                "type": "upstream_http_error",
                "code": response.status_code if response is not None else 500,
            }
        }
        yield {"data": json.dumps(error_chunk, ensure_ascii=False)}
        yield {"data": "[DONE]"}
        return
    except httpx.RequestError as exc:
        error_chunk = {
            "error": {
                "message": f"请求上游失败: {str(exc)}",
                "type": "upstream_request_error",
                "code": 502,
            }
        }
        yield {"data": json.dumps(error_chunk, ensure_ascii=False)}
        yield {"data": "[DONE]"}
        return
    except Exception as exc:  # noqa: BLE001
        error_chunk = {
            "error": {
                "message": f"流式转换失败: {str(exc)}",
                "type": "stream_proxy_error",
                "code": 500,
            }
        }
        yield {"data": json.dumps(error_chunk, ensure_ascii=False)}
        yield {"data": "[DONE]"}
        return

    # 流式结束时输出 finish_reason=stop 的结束 chunk，再发送 [DONE]
    yield {"data": json.dumps(_build_chat_stream_chunk(response_id, model, {}, "stop"), ensure_ascii=False)}
    yield {"data": "[DONE]"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": {"message": f"非法 JSON 请求体: {str(exc)}"}})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "请求体必须是 JSON 对象"}})

    if not body.get("model"):
        return JSONResponse(status_code=400, content={"error": {"message": "缺少必填字段 model"}})

    if not isinstance(body.get("messages"), list):
        return JSONResponse(status_code=400, content={"error": {"message": "messages 必须是数组"}})

    payload = _build_upstream_payload(body)
    model = body.get("model")
    is_stream = bool(body.get("stream", False))

    if is_stream:
        return EventSourceResponse(_stream_upstream_as_chat_completions(payload, model))

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            upstream_response = await client.post(
                f"{UPSTREAM_API_URL.rstrip('/')}/responses",
                headers=_make_headers(),
                json=payload,
            )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"请求上游失败: {str(exc)}", "type": "upstream_request_error"}},
        )

    if upstream_response.status_code >= 400:
        return _error_response_from_upstream(
            upstream_response.status_code,
            upstream_response.text,
            upstream_response.headers.get("content-type"),
        )

    try:
        upstream_json = upstream_response.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "上游返回的不是合法 JSON", "type": "invalid_upstream_json"}},
        )

    return JSONResponse(content=_responses_to_chat_completion(upstream_json, model))


if __name__ == "__main__":
    _cleanup_previous_gateway_process()
    uvicorn.run(app, host=HOST, port=PORT)
