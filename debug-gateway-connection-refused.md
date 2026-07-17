# Debug Session: gateway-connection-refused
- **Status**: [OPEN]
- **Issue**: Trae 访问 `http://127.0.0.1:8000/v1/chat/completions` 时偶发 `Connection refused`
- **Debug Server**: N/A
- **Log File**: user-journalctl

## Reproduction Steps
1. 通过 Trae 访问本地网关 `http://127.0.0.1:8000/v1/chat/completions`
2. 客户端报错：`Connection refused`

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | 服务没有启动成功 | Medium | Low | Rejected：`systemctl --user show` 显示 `active/running` |
| B | 8000 端口没有监听 | Medium | Low | Rejected：`ss -ltnp` 显示监听存在 |
| C | 服务周期性优雅退出，导致端口空窗 | High | Low | Confirmed：`journalctl --user` 显示每 7-8 秒 shutdown 后被 systemd 重启 |
| D | 用户级 systemd 因服务启动方式不合适导致主进程自退出 | High | Medium | Pending |

## Log Evidence
- `systemctl --user show vln-api-gateway.service`：`ActiveState=active`、`SubState=running`
- `ss -ltnp '( sport = :8000 )'`：确认 `127.0.0.1:8000` 存在监听
- `journalctl --user -u vln-api-gateway.service -n 80`：
  - `Started server process [...]`
  - 约 3-4 秒后出现 `Shutting down`
  - 随后 `Finished server process [...]`
  - systemd 记录 `Scheduled restart job`

## Verification Conclusion
最终确认根因是“手动启动的残留进程”和“用户级 systemd 服务”同时争抢 `127.0.0.1:8000`：

- 残留手动进程：
  - `/home/wzy/anaconda3/envs/vln/bin/python /home/wzy/vln_ws/tools/api_gateway.py`
- 用户级 systemd 服务：
  - `/home/wzy/anaconda3/envs/vln/bin/python -m uvicorn tools.api_gateway:app --host 127.0.0.1 --port 8000`

并发后果：
- systemd 进程抢不到端口时记录 `[Errno 98] address already in use`
- 因配置了 `Restart=always`，service 进入 `auto-restart`
- 客户端在重启空窗期访问时，就会得到 `Connection refused`

当前处理：
- 已清理残留手动进程
- 保留单一的用户级 systemd 服务作为正式入口
