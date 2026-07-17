# Debug Session: remote-ssh-model-fail

Status: OPEN

## Symptom

- From another computer over SSH, calling the model fails.
- Opening the local project and calling the same model on this machine succeeds.

## Initial Hypotheses

- H1: The gateway only listens on `127.0.0.1`, so remote machines cannot reach it even though local requests work.
- H2: The remote caller uses a different base URL, port, or model name than the successful local project.
- H3: The gateway process reached by the remote caller is not the current fixed version of `api_gateway.py`.
- H4: An environment difference on the remote path strips or changes required headers such as `Authorization`.
- H5: The failure happens before model execution, such as firewall, bind address, or reverse proxy routing.

## Evidence Plan

- Read the handoff document and current gateway implementation.
- Add request-path instrumentation only after confirming the likely observation points.
- Reproduce with one successful local request and one failing remote-style request.
- Compare bind address, request headers, target model mapping, and upstream call results.

## Runtime Evidence

- `systemctl --user status vln-api-gateway.service --no-pager`
  - Service is active and running.
  - Main process is `python -m uvicorn tools.api_gateway:app --host 127.0.0.1 --port 8000`.
- `ss -ltnp '( sport = :8000 )'`
  - Listener is only `127.0.0.1:8000`.
  - No listener is bound on `0.0.0.0:8000` or the machine LAN IP.
- `hostname -I`
  - Machine has reachable non-loopback addresses such as `192.168.31.234`.
- Local request test:
  - `curl http://127.0.0.1:8000/v1/chat/completions` returns `200` and valid model output.
- Non-loopback request test:
  - `curl http://192.168.31.234:8000/v1/chat/completions` fails with `Could not connect to server`.

## Hypothesis Status

- H1: Confirmed. The gateway is bound only to loopback, so only local requests can reach it.
- H2: Rejected for this symptom. The same model succeeds locally using the current gateway.
- H3: Rejected. The active process is the expected systemd-managed uvicorn instance.
- H4: Not needed to explain the current failure because the TCP connection fails before request forwarding.
- H5: Confirmed at network/bind layer. Failure happens before model execution.

## Current Conclusion

- The issue is not a model or protocol-conversion regression.
- The current deployment is intentionally local-only because both code defaults and systemd startup use `127.0.0.1`.
- If another machine needs to call this gateway directly, the service must bind to a non-loopback address such as `0.0.0.0`, or the caller must use SSH local port forwarding.

## Fix Applied

- Updated user-level systemd service to read `HOST` and `PORT` from the environment file.
- Changed `/home/wzy/.config/vln-api-gateway/gateway.env` from `HOST=127.0.0.1` to `HOST=0.0.0.0`.
- Updated `/home/wzy/vln_ws/tools/install_vln_api_gateway_systemd.sh` so future system-level installs also use `HOST` and `PORT` from the environment file.

## Post-Fix Verification

- `systemctl --user restart vln-api-gateway.service`
  - Service restarts successfully.
- `systemctl --user status vln-api-gateway.service --no-pager -l`
  - Main process now runs with `--host 0.0.0.0 --port 8000`.
- `ss -ltnp '( sport = :8000 )'`
  - Listener is now `0.0.0.0:8000`.
- `curl http://127.0.0.1:8000/v1/chat/completions`
  - Returns `200`.
- `curl http://192.168.31.234:8000/v1/chat/completions`
  - Returns `200`.

## Additional Evidence After User Retry

- User-reported error still targets `http://127.0.0.1:8000/v1/chat/completions`.
- Re-checked gateway host state:
  - `systemctl --user status vln-api-gateway.service --no-pager -l` shows the service is active and running with `--host 0.0.0.0 --port 8000`.
  - `ss -ltnp '( sport = :8000 )'` shows `LISTEN 0.0.0.0:8000`.
  - Local server journal contains successful requests from both `127.0.0.1` and `192.168.31.234`.
- Therefore the current failure is no longer "gateway not reachable on the server host".
- The active root cause is now client-side endpoint selection:
  - `127.0.0.1` from the other computer means "that other computer itself", not this gateway machine.

## Updated Conclusion

- Server-side bind issue has been fixed.
- Remaining problem is that the remote caller still points to `127.0.0.1:8000` instead of the gateway machine address or an SSH-forwarded local port.
