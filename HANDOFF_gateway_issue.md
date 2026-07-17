# 网关问题交接文档

## 1. 项目背景

当前项目在本地提供一个 OpenAI Chat Completions 兼容网关：

- 对外入口：`http://127.0.0.1:8000/v1/chat/completions`
- 上游协议：OpenAI Responses API
- 当前主要实现文件：`/home/wzy/vln_ws/tools/api_gateway.py`

最近处理过两类问题：

1. **协议转换问题**
   - 在切换到 `gpt-5.6-sol` 这类走 Responses API 的模型后，复杂工具历史消息会失败。
   - 已修复 `assistant.tool_calls`、`role="tool"`、`tools`、`tool_choice` 等转换逻辑。

2. **本地连接拒绝问题**
   - Trae 偶发报错：
     - `Connection refused`
     - `WebSocket error: ... tcp connect error ... 127.0.0.1:8000`
   - 这是本交接文档重点。

## 2. 本次问题现象

Trae 访问本地网关时，偶发出现：

```text
WebSocket error: Network IO error: ...
http://127.0.0.1:8000/v1/chat/completions
ConnectionRefused
```

表现特征：

- 有时候请求能成功
- 有时候请求刚好命中空窗期，会报 `Connection refused`
- 不是持续性“服务完全起不来”

## 3. 已确认根因

根因已经确认，不是模型协议转换本身，而是**两套启动方式同时存在，争抢同一个 8000 端口**。

### 3.1 冲突的两套启动方式

#### A. 手动启动残留进程

命令形态：

```bash
/home/wzy/anaconda3/envs/vln/bin/python /home/wzy/vln_ws/tools/api_gateway.py
```

#### B. 用户级 systemd 服务

服务文件：

- `/home/wzy/.config/systemd/user/vln-api-gateway.service`

当前 `ExecStart`：

```bash
/home/wzy/anaconda3/envs/vln/bin/python -m uvicorn tools.api_gateway:app --host 127.0.0.1 --port 8000
```

### 3.2 冲突后的后果

当手动进程还占着 `127.0.0.1:8000` 时，用户级 systemd 服务启动会失败：

- journalctl 中出现：

```text
ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use
```

- 由于服务配置里有：

```ini
Restart=always
RestartSec=3
```

所以它会持续 `auto-restart`。

结果就是：

- 有一个服务在不断重试重启
- 端口在某些时间点会出现监听空窗
- Trae 在空窗期访问时，就得到 `Connection refused`

## 4. 已完成处理

### 4.1 已做的代码/配置调整

1. **清理了网关脚本中的调试日志**
   - 删除了完整请求体、转换 payload、SSE 事件摘要等调试打印
   - 保留了必要的错误返回逻辑

2. **保留用户级 systemd 作为正式入口**
   - 服务文件：`/home/wzy/.config/systemd/user/vln-api-gateway.service`

3. **把 systemd 的启动方式改成模块加载**
   - 从：

```bash
python /home/wzy/vln_ws/tools/api_gateway.py
```

   - 改成：

```bash
python -m uvicorn tools.api_gateway:app --host 127.0.0.1 --port 8000
```

这样可以避免走脚本的 `__main__` 路径，作为服务进程更稳定。

4. **准备了系统级安装脚本**
   - 文件：`/home/wzy/vln_ws/tools/install_vln_api_gateway_systemd.sh`

### 4.2 已做的运行修复

1. 清理了残留手动进程
2. 重启了用户级 systemd 服务
3. 确认当前监听进程已变为：

```text
/home/wzy/anaconda3/envs/vln/bin/python -m uvicorn tools.api_gateway:app --host 127.0.0.1 --port 8000
```

而不是旧的手动 `api_gateway.py` 进程

## 5. 当前关键文件

### 核心代码

- `/home/wzy/vln_ws/tools/api_gateway.py`

### 用户级 systemd 服务

- `/home/wzy/.config/systemd/user/vln-api-gateway.service`

### 用户级环境变量文件

- `/home/wzy/.config/vln-api-gateway/gateway.env`

### 系统级安装脚本

- `/home/wzy/vln_ws/tools/install_vln_api_gateway_systemd.sh`

### 本次问题调试记录

- `/home/wzy/vln_ws/debug-gateway-connection-refused.md`

## 6. 当前推荐运行方式

后续**不要再手动运行**：

```bash
/home/wzy/anaconda3/envs/vln/bin/python /home/wzy/vln_ws/tools/api_gateway.py
```

统一只用 `systemd` 管理。

### 用户级 systemd 常用命令

```bash
systemctl --user status vln-api-gateway.service
systemctl --user restart vln-api-gateway.service
systemctl --user stop vln-api-gateway.service
systemctl --user start vln-api-gateway.service
```

### 检查是否有手动残留进程

```bash
ps -eo pid,ppid,cmd | grep -E 'api_gateway.py|uvicorn tools.api_gateway:app' | grep -v grep
```

理想状态应只剩下 `uvicorn tools.api_gateway:app` 这一条。

## 7. 当前验证结果

在修复冲突后，已通过本机请求验证：

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","stream":false,"messages":[{"role":"user","content":"你好，回复一个ok"}]}'
```

返回正常 `ok`。

## 8. 尚未完全闭环的点

### 8.1 真正“开机即启动”还差一步

当前已经配置好**用户级** systemd，但机器目前：

```text
Linger=no
```

这意味着：

- 登录后会自动拉起用户级服务
- 但不是“系统刚开机、用户未登录时”就自动运行

若要真正做到开机级常驻，需要用户本机执行：

```bash
sudo loginctl enable-linger wzy
sudo /home/wzy/vln_ws/tools/install_vln_api_gateway_systemd.sh
```

### 8.2 需要持续坚持“单一入口”

如果后续再次手动运行 `python tools/api_gateway.py`，还是可能复现同类端口冲突问题。

## 9. 给新对话的建议起手式

如果新对话需要继续接手，建议先做这几步：

1. 先确认当前监听进程是谁：

```bash
ss -ltnp '( sport = :8000 )'
ps -eo pid,ppid,cmd | grep -E 'api_gateway.py|uvicorn tools.api_gateway:app' | grep -v grep
```

2. 再确认用户级服务状态：

```bash
systemctl --user status vln-api-gateway.service --no-pager
journalctl --user -u vln-api-gateway.service -n 50 --no-pager
```

3. 如果又出现 `Connection refused`，优先排查：
   - 是否又有手动残留进程
   - systemd 服务是否在抢端口失败后 auto-restart
   - 是否存在新的端口占用者

## 10. 一句话总结

这次 `Connection refused` 的根因是：

**手动启动残留进程 + 用户级 systemd 服务同时争抢 8000 端口，导致 systemd 反复重启并产生端口空窗。**

正式解决策略是：

**以后只保留 systemd 作为唯一入口，不再手动运行 `tools/api_gateway.py`。**
