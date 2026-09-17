# 把扩散模型部署成 HTTP 服务（DRIVE Thor / stable-diffusion.cpp）

> One-liner: How to deploy a stable-diffusion.cpp model as an HTTP service on an
> NVIDIA DRIVE Thor board — prerequisites, upload, systemd installation, self-checks,
> troubleshooting, and a complete single-file implementation using only the Python
> standard library. A short closing section compares on-demand vs resident deployment
> as a **recommendation only** — either is workable.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264 / sm_101a），DriveOS 7.0.3，CUDA 12.8
> 本仓库版：脱敏整理

---

> **单位口径（本文档统一）**
> - **GiB** 用于池／显存口径（本平台大页池 46 GiB 级；`sd.cpp` 打印的 "MB" 实际是 **MiB**）
> - **GB** 用于磁盘文件口径（1 GB = 10⁹ B）
> - 例：H3 权重磁盘占 **45.0 GB**（= 41.9 GiB），运行时占 VRAM **42.0 GiB**（框架报 `43 017 MB`）

## 0. 这份文档讲什么

把已经能在板上命令行出图的扩散模型（`sd-cli`）**包装成一个 HTTP 服务**，让局域网里的
其他程序 / AI / Agent 直接调用，不必登录板子。

- **第 1–2 节 = 部署步骤**（照抄即可）：前置检查 → 放文件 → 校验 → 装 systemd → 自检；
- **第 3–7 节 = 实现与使用**：完整代码、接口速查、调用注意事项、实测、排障；
- **第 8 节 = 常驻还是按需**：这是**建议**，不是必须——两种都行，按你的场景选。

> 如果你还没有二进制，可直接用 [prebuilt-binary-quickstart.md](../05-prebuilt-binary/prebuilt-binary-quickstart.md)
> 里的预编译产物（下载即用，含校验值），省掉交叉编译。

---

## 1. 前置条件（部署前逐项确认）

| 项 | 要求 | 检查命令 |
|---|---|---|
| 二进制 | `bin/sd-cli`（aarch64 / sm_101，来自本项目交叉编译或预编译包） | `file bin/sd-cli` |
| 模型文件 | DiT + VAE + 文本编码器三件套（Z-Image） | `ls -l models/` |
| Python | 板上自带 python3（**无需 pip**，实现只用标准库） | `python3 -V` |
| 权限 | 能 `sudo`（装 systemd 单元） | `sudo -n true && echo OK` |
| 端口 | 默认 8081 未被占用 | `ss -ltnp \| grep 8081 \|\| echo free` |
| 大页池 | 池余量足够跑扩散模型（见 [`../05-system-tuning/hugepage-pool.md`](../02-h3-video-audio/h3-video-audio.md#一内存预算h3-为什么必须独占池)） | `cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages` |

目录约定（下同）：

```
<DATA_DIR>/sd-cpp/
├── bin/sd-cli, bin/sd-server
├── models/{z_image_turbo-*.gguf, ae.safetensors, Qwen3-4B-Q4_K_M.gguf}
├── out/                     产物
├── api.log                  每次出图的耗时记录
└── zimage_api.py            本服务
```

---

## 2. 部署步骤

### 2.1 放置文件

```bash
sudo mkdir -p <DATA_DIR>/sd-cpp/{bin,models,out}
sudo chown -R "$USER" <DATA_DIR>/sd-cpp

# 二进制与模型（板上无 rsync，用 scp -o Compression=no；传完逐个核对 sha256）
scp bin/sd-cli bin/sd-server  <board>:<DATA_DIR>/sd-cpp/bin/
scp models/**                 <board>:<DATA_DIR>/sd-cpp/models/
scp zimage_api.py             <board>:<DATA_DIR>/sd-cpp/
ssh <board> 'chmod 755 <DATA_DIR>/sd-cpp/bin/sd-* <DATA_DIR>/sd-cpp/zimage_api.py'
```

### 2.2 校验（装之前先验，别等 systemd 报错）

```bash
ssh <board> 'cd <DATA_DIR>/sd-cpp
  python3 -m py_compile zimage_api.py && echo "  python 语法 OK"
  ./bin/sd-cli --help >/dev/null && echo "  二进制可执行 OK"
  ldd ./bin/sd-cli | grep -c "not found" | sed "s/^/  缺失依赖数: /"'
```

`ldd` 缺失数为 0 即可（CUDA 运行库随 DriveOS 提供；若用了带 CUDA 运行库的预编译包，按包内说明设 `LD_LIBRARY_PATH`）。

### 2.3 安装 systemd 单元并启动

```bash
# 单元文件见 scripts/zimage-api.service（按需改 User / 路径 / 端口）
scp scripts/zimage-api.service <board>:/tmp/
ssh <board> '
  sudo install -m 644 /tmp/zimage-api.service /etc/systemd/system/zimage-api.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now zimage-api
  systemctl is-active zimage-api && systemctl is-enabled zimage-api'
```

### 2.4 自检（4 条，全过才算部署成功）

```bash
ssh <board> 'curl -s --max-time 5 localhost:8081/health; echo
             curl -s localhost:8081/v1/models; echo
             sudo journalctl -u zimage-api -n 10 --no-pager | tail -5'
```

再加一次**真实出图**（最小尺寸，几十秒）：

```bash
curl -s -X POST localhost:8081/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table, studio light","size":"512x512","steps":8,"response_format":"path"}'
# 期望：data[0].path 有值，meta.wall_s 有数字，out/ 下出现 png
```

### 2.5 确认没有影响同机已运行的服务

若同机已有常驻 LLM 服务在跑（本平台的常见形态），装完必须复核它没被影响：

```bash
ssh <board> 'systemctl is-active <llm-service>            # 仍 active
             curl -s localhost:8080/health                # 服务自身健康
             cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages'  # 池余量回到装前水平
```

**服务空闲时应为 0 显存占用**——这是本实现的关键特性（见 §8）。

---

## 3. 完整实现（单文件、标准库）

> 仓库内同路径脚本：`scripts/zimage-api.py`
> 环境变量：`PORT`(8081) `SD_DIR` `ZIMG` `DEFAULT_STEPS`(8) `DEFAULT_SIZE`(1024)
> `CFG_SCALE`(1.0) `JOB_TIMEOUT`(900)

```python
#!/usr/bin/env python3
"""按需 HTTP API（stable-diffusion.cpp 封装）· v2（面向 AI 调用者优化）

v2 相对 v1 的变化：
  1) 输出格式可选取：response_format = "b64_json"(默认,OpenAI 兼容) | "url" | "path"
     —— **给 LLM/Agent 用必须选 url 或 path**：一张 1024² PNG 的 base64 约 60 万 token，
        塞进模型上下文会直接撑爆窗口。
  2) 支持异步：async=true 立即返回 job_id + poll_url，避免 Agent 的工具调用超时
     （重载时出图要 100+ s，多数 Agent 的工具超时在 60–120 s）。
  3) 新增 GET /files/<name> 提供产物直链（仅限 out/ 目录下的 png）。
  4) GET /v1/jobs/<id> 查询异步任务；GET /jobs 列最近任务。

设计要点：
  · 空闲时不占任何显存：收请求 → 拉起 sd-cli 出图 → 进程退出、显存归还。
  · 单工作线程串行（只有一块 GPU），请求进队列，可查排队深度。
"""
import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SD_DIR = os.environ.get("SD_DIR", "/brand_data/sd-cpp")
SD_BIN = os.path.join(SD_DIR, "bin", "sd-cli")
MODELS = os.path.join(SD_DIR, "models")
OUT_DIR = os.path.join(SD_DIR, "out")
PORT = int(os.environ.get("PORT", "8081"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "8"))
DEFAULT_SIZE = int(os.environ.get("DEFAULT_SIZE", "1024"))
CFG_SCALE = os.environ.get("CFG_SCALE", "1.0")
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))
LOG_PATH = os.path.join(SD_DIR, "api.log")
HP = "/sys/kernel/mm/hugepages/hugepages-2048kB"
MAX_JOBS_KEPT = 60


def pick_weights():
    env = os.environ.get("ZIMG")
    if env:
        return env
    q8 = os.path.join(MODELS, "z_image_turbo-Q8_0.gguf")
    return q8 if os.path.exists(q8) else os.path.join(MODELS, "z_image_turbo-Q4_K_S.gguf")


WEIGHTS = pick_weights()
VAE = os.path.join(MODELS, "ae.safetensors")
TE = os.path.join(MODELS, "Qwen3-4B-Q4_K_M.gguf")

JOBS_Q = queue.Queue()
DB = {}                      # job_id -> 状态字典
DB_ORDER = []                # 保留最近 MAX_JOBS_KEPT 个
LOCK = threading.Lock()
STATS = {"done": 0, "failed": 0, "busy": False, "current": None, "last": None}


def log(msg):
    line = "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass
    print(line.rstrip(), flush=True)


def pool():
    try:
        total = int(open(os.path.join(HP, "nr_hugepages")).read().strip())
        free = int(open(os.path.join(HP, "free_hugepages")).read().strip())
        return {"total_pages": total, "free_pages": free,
                "free_gib": round(free * 2 / 1024, 2)}
    except Exception as e:
        return {"error": str(e)}


def parse_size(s):
    if not s:
        return DEFAULT_SIZE, DEFAULT_SIZE
    s = str(s).lower().replace("*", "x")
    if "x" in s:
        w, h = s.split("x")[:2]
        return int(w), int(h)
    v = int(s)
    return v, v


def run_job(job):
    """执行一次出图，返回 (ok, out_path, meta)"""
    t0 = time.time()
    out_path = os.path.join(OUT_DIR, "api_%s.png" % job["id"])
    cmd = [SD_BIN, "--diffusion-model", WEIGHTS, "--vae", VAE, "--llm", TE,
           "--cfg-scale", CFG_SCALE, "--steps", str(job["steps"]), "--diffusion-fa",
           "-W", str(job["width"]), "-H", str(job["height"]),
           "-p", job["prompt"], "-o", out_path]
    if job.get("seed") is not None:
        cmd += ["--seed", str(job["seed"])]
    if job.get("negative"):
        cmd += ["-n", job["negative"]]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + \
        ":/usr/local/cuda-12.8/targets/aarch64-linux/lib"
    ok, gen_s, err = False, None, None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT,
                           env=env, cwd=SD_DIR)
        for ln in (r.stdout or "").splitlines():
            if "generate_image completed in" in ln:
                try:
                    gen_s = float(ln.split("completed in")[1].split("s")[0].strip())
                except Exception:
                    pass
            if "save result image" in ln and "success" in ln:
                ok = True
        if r.returncode != 0:
            ok = False
            err = ((r.stdout or "")[-900:] + (r.stderr or "")[-400:])
        elif not os.path.exists(out_path):
            ok, err = False, "sd-cli 返回 0 但无产物文件"
    except subprocess.TimeoutExpired:
        err = "生成超时（%ss）" % JOB_TIMEOUT
    wall = round(time.time() - t0, 2)
    meta = {"job_id": job["id"], "wall_s": wall, "generate_s": gen_s,
            "width": job["width"], "height": job["height"], "steps": job["steps"],
            "weights": os.path.basename(WEIGHTS), "queue_wait_s": job.get("queue_wait_s"),
            "path": out_path if ok else None,
            "url": "/files/%s" % os.path.basename(out_path) if ok else None}
    if err:
        meta["error"] = err.replace("\n", " | ")[:600]
    log("job=%s ok=%s gen=%ss wall=%ss %dx%d steps=%d qwait=%ss" % (
        job["id"], ok, gen_s, wall, job["width"], job["height"], job["steps"],
        job.get("queue_wait_s")))
    return ok, out_path if ok else None, meta


def worker():
    while True:
        job = JOBS_Q.get()
        # 真实排队时长：在任务被 worker 取到、真正开始执行时计算（不是入队瞬间）
        job["queue_wait_s"] = round(time.time() - job.get("enqueued", time.time()), 2)
        with LOCK:
            STATS["busy"] = True
            STATS["current"] = job["id"]
            DB[job["id"]]["status"] = "running"
            DB[job["id"]]["started"] = time.strftime("%H:%M:%S")
        ok, path, meta = run_job(job)
        with LOCK:
            STATS["busy"] = False
            STATS["current"] = None
            STATS["done" if ok else "failed"] += 1
            STATS["last"] = meta
            DB[job["id"]].update({"status": "done" if ok else "failed", "meta": meta,
                                  "path": path, "finished": time.strftime("%H:%M:%S")})
        job["result"] = (ok, path, meta)
        job["event"].set()
        JOBS_Q.task_done()


threading.Thread(target=worker, daemon=True).start()


def shape_payload(job, ok, path, meta):
    """按请求时选定的 response_format 组织返回体（元数据永远带 meta）"""
    fmt = job.get("response_format", "b64_json")
    item = {"revised_prompt": job["prompt"]}
    if fmt == "path":
        item["path"] = path
    elif fmt == "url":
        item["url"] = "/files/%s" % os.path.basename(path or "")
    else:
        try:
            with open(path, "rb") as f:
                item["b64_json"] = base64.b64encode(f.read()).decode()
        except Exception as e:
            item["b64_json"] = None
            meta["error"] = "读产物失败: %s" % e
    return {"created": int(time.time()), "model": "z-image-turbo",
            "data": [item], "meta": meta}


class Handler(BaseHTTPRequestHandler):
    server_version = "img-api/2.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_file(self, path):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            self._send(404, {"error": {"message": "文件不存在"}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except Exception:
            return {}

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self._send(200, {"status": "ok", "busy": STATS["busy"],
                             "queue_depth": JOBS_Q.qsize(),
                             "model": os.path.basename(WEIGHTS), "pool": pool()})
        elif p == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "z-image-turbo", "object": "model", "owned_by": "local",
                 "weights": os.path.basename(WEIGHTS),
                 "defaults": {"size": DEFAULT_SIZE, "steps": DEFAULT_STEPS},
                 "response_formats": ["b64_json", "url", "path"],
                 "async_supported": True}]})
        elif p == "/status":
            with LOCK:
                recent = [{"job_id": k, "status": DB[k]["status"], "meta": DB[k].get("meta")}
                          for k in DB_ORDER[-5:]]
            self._send(200, {"state": STATS, "pool": pool(), "weights": WEIGHTS,
                             "out_dir": OUT_DIR, "recent_jobs": recent})
        elif p == "/jobs":
            with LOCK:
                recent = [{"job_id": k, "status": DB[k]["status"]} for k in DB_ORDER[-20:]]
            self._send(200, {"jobs": recent, "queue_depth": JOBS_Q.qsize()})
        elif p.startswith("/v1/jobs/"):
            jid = p[len("/v1/jobs/"):]
            with LOCK:
                j = DB.get(jid)
            if not j:
                self._send(404, {"error": {"message": "未知任务 %s" % jid}})
            elif j["status"] in ("queued", "running"):
                self._send(200, {"job_id": jid, "status": j["status"],
                                 "queue_depth": JOBS_Q.qsize()})
            else:
                ok = j["status"] == "done"
                self._send(200 if ok else 500,
                           shape_payload(j["job"], ok, j.get("path"), j.get("meta") or {}))
        elif p.startswith("/files/"):
            name = os.path.basename(p[len("/files/"):])
            if not re.fullmatch(r"[A-Za-z0-9_\-]+\.png", name):
                self._send(400, {"error": {"message": "非法文件名"}})
                return
            self._send_file(os.path.join(OUT_DIR, name))
        else:
            self._send(404, {"error": {"message": "not found: %s" % p}})

    def do_POST(self):
        p = self.path.split("?")[0]
        if p not in ("/v1/images/generations", "/sdapi/v1/txt2img", "/v1/img_gen"):
            self._send(404, {"error": {"message": "unsupported endpoint: %s" % p}})
            return
        body = self._read_json()
        prompt = body.get("prompt") or body.get("input") or ""
        if not prompt:
            self._send(400, {"error": {"message": "prompt 不能为空"}})
            return
        w, h = parse_size(body.get("size") or body.get("resolution"))
        fmt = body.get("response_format")
        if fmt not in ("b64_json", "url", "path"):
            fmt = "b64_json"
        job = {"id": uuid.uuid4().hex[:12], "prompt": prompt,
               "negative": body.get("negative_prompt") or body.get("n_prompt"),
               "width": int(body.get("width") or w), "height": int(body.get("height") or h),
               "steps": int(body.get("steps") or DEFAULT_STEPS),
               "seed": body.get("seed"), "response_format": fmt,
               "event": threading.Event(), "enqueued": time.time()}
        with LOCK:
            DB[job["id"]] = {"status": "queued", "job": job,
                             "submitted": time.strftime("%H:%M:%S")}
            DB_ORDER.append(job["id"])
            while len(DB_ORDER) > MAX_JOBS_KEPT:
                DB.pop(DB_ORDER.pop(0), None)
        JOBS_Q.put(job)   # queue_wait_s 由 worker 在真正开始执行时计算（见 worker()）

        if body.get("async") or body.get("async_mode"):
            self._send(202, {"job_id": job["id"], "status": "queued",
                             "poll_url": "/v1/jobs/%s" % job["id"],
                             "queue_depth": JOBS_Q.qsize()})
            return

        if not job["event"].wait(JOB_TIMEOUT + 30):
            self._send(504, {"error": {"message": "等待超时", "job_id": job["id"]}})
            return
        ok, path, meta = job["result"]
        if p == "/sdapi/v1/txt2img":
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            except Exception:
                b64 = None
            if ok:
                self._send(200, {"images": [b64], "info": json.dumps(meta)})
            else:
                self._send(500, {"images": [], "info": json.dumps(meta)})
            return
        if not ok:
            self._send(500, {"error": {"message": "生成失败", "detail": meta.get("error"),
                                       "job_id": job["id"]}})
            return
        self._send(200, shape_payload(job, ok, path, meta))

    def log_message(self, fmt, *args):
        pass

    def log_error(self, *args):
        pass


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    log("img-api v2 启动：port=%d weights=%s 池余=%s" %
        (PORT, os.path.basename(WEIGHTS), pool()))
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
```

### systemd 单元

```ini
# scripts/zimage-api.service
[Unit]
Description=Diffusion on-demand HTTP API (stable-diffusion.cpp wrapper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=user
Group=user
WorkingDirectory=/brand_data/sd-cpp
Environment=PORT=8081
Environment=SD_DIR=/brand_data/sd-cpp
ExecStart=/usr/bin/python3 /brand_data/sd-cpp/zimage_api.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
# 注意：不要设置 MemoryMax/MemoryHigh —— 出图进程的显存来自大页池，
# 加内存上限可能把它误杀（设计时已考虑，故不设）。

[Install]
WantedBy=multi-user.target
```

---

## 4. 接口速查

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活 + `busy` + `queue_depth` + **池余量**（调用前先看这个） |
| GET | `/v1/models` | 模型列表（默认尺寸/步数、支持的 `response_format`、是否支持 async） |
| GET | `/status` | 详细状态：当前任务、池、权重、累计成功/失败、最近 5 个任务 |
| GET | `/jobs`、`/v1/jobs/<id>` | 任务列表 / 单任务查询（异步模式轮询用） |
| GET | `/files/<name>` | 产物直链（仅 `out/` 下的 png，文件名白名单校验） |
| POST | `/v1/images/generations` | **OpenAI 兼容**（主力接口） |
| POST | `/sdapi/v1/txt2img` | **A1111 兼容**（返回 `{images:[b64], info:"{...}"}`） |
| POST | `/v1/img_gen` | 同上（兼容别名） |

请求示例（异步 + 只回路径，**Agent 场景推荐**）：

```bash
curl -s -X POST localhost:8081/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table, studio light",
       "size":"1024x1024","steps":8,"async":true,"response_format":"path"}'
# → 202 {"job_id":"0c63e7f4295c","status":"queued","poll_url":"/v1/jobs/0c63e7f4295c","queue_depth":1}

curl -s localhost:8081/v1/jobs/0c63e7f4295c
# → 200 {"created":...,"data":[{"path":"/brand_data/sd-cpp/out/api_0c63e7f4295c.png"}],
#        "meta":{"wall_s":58.3,"generate_s":57.1,"width":1024,"height":1024,"steps":8,"queue_wait_s":0.0}}
```

### `response_format` 三档

| 取值 | 返回 | 用途 |
|---|---|---|
| `b64_json`（默认） | `data[0].b64_json` | 传统 OpenAI 客户端 |
| `url` | `data[0].url` = `/files/<name>` | 让调用方自己取图 |
| `path` | `data[0].path` = 板上绝对路径 | **AI / Agent 首选** |

---

## 5. 给 AI / Agent 调用者的三条硬规则

这是本实现最容易被忽略、但**不设计就会被坑死**的部分。

### ① 千万别把图像数据塞回模型上下文

一张 1024×1024 PNG ≈ 1.7 MB，base64 后 ≈ 2.3 MB 字符，折合 **约 60 万 token**——
直接爆掉上下文窗口。

⇒ 用 `response_format: path|url`，让 Agent 只拿到一个几十字节的字符串；
**需要模型判断画面时，再走视觉接口单独看图**。

### ② 工具调用一律走异步，不要同步等

重载时单张图 100+ s，而多数 Agent 框架的工具超时在 60–120 s ⇒ 同步调用会被掐断并触发重试
（越重试越堵）。

⇒ 用 `async: true` 拿 `job_id`，然后轮询 `/v1/jobs/<id>`（间隔 5–10 s）。

### ③ 预期时延是"叠加"而非"并行"

一块 GPU 上，Agent 的 LLM 轮次与扩散去噪**互抢算力且都串行**：

| 环节 | 典型耗时（本机实测口径） |
|---|---|
| Agent 的 LLM 第 1 轮（判读 + 发起工具调用，3–5 万 token 上下文） | 60–110 s |
| 出图（1024² / 8 步） | 57 s（LLM 空闲）～110 s（LLM 重载） |
| Agent 的 LLM 第 2 轮（工具结果入上下文） | 60–110 s |
| **单张图端到端** | **约 3–6 分钟**；多轮自检/多图线性叠加 |

⇒ 同机三件套（Agent + LLM + 扩散）能跑通，但要按"分钟级"设计交互：
异步、只回路径、超时放宽（HTTP 客户端建议 ≥300 s）。这也说明同机部署更适合
**批处理 / 后台出图**，不适合"秒回"的交互。

---

## 6. 实测性能（2026-09-17，与同机常驻 LLM 服务并存）

**时延取决于同机 LLM 的负载——报数必须带口径**：

| 场景 | 1024² / 8 步 | 512² / 8 步 |
|---|---|---|
| LLM 服务**空闲** | ≈ **57 s** | ≈ **13.5 s** |
| LLM 服务**轻载**（并发极短请求，如 16 token 探测） | ≈ 59.8 s（+4%） | — |
| LLM 服务**重载**（正在跑 3 万+ token 的长上下文） | ≈ **108–110 s**（约 2×） | ≈ 28 s |

- 空闲时池余量回到 **6054 页（11.8 GiB）** ⇒ 服务本身**不占显存**
- **远程 HTTP 调用与板上直连命令行耗时一致**（慢是因为同机 LLM 在跑，不是网络层）
- 出图前后池空闲页一致 ⇒ **无泄漏**

---

## 7. 排障清单

| 症状 | 根因 | 处理 |
|---|---|---|
| `curl` 无响应 / 连接被拒 | 服务没起或端口占用 | `systemctl status zimage-api`；`ss -ltnp \| grep 8081` |
| `/health` 里 `pool.free_gib` 很小 | 同机服务占着池 | 停占用方（池机制见 [主文档 §一 内存预算](../02-h3-video-audio/h3-video-audio.md#一内存预算h3-为什么必须独占池)），或等它空闲 |
| 出图报 OOM（`cudaMalloc failed`） | 池余量不足 | 别清池；停占池服务或降分辨率/量化档 |
| `/v1/jobs/<id>` 一直 `queued` | 前面任务在跑（串行队列） | 看 `/health` 的 `queue_depth`；等或加机器 |
| Agent 工具调用频繁超时/重试 | 用了同步等待 | 改 `async: true` + 轮询 |
| 上下文被一张图撑爆 | 返回了 `b64_json` | 用 `response_format: path` |
| 想出 1024² 以上（hires）失败 | 池余量不足（先崩在 VAE） | 保持原生分辨率；或先腾池 |
| systemd 把出图进程杀掉 | 设了 `MemoryMax`/`MemoryHigh` | **不要设**——它的显存来自大页池 |
| 板端装不了 `fastapi`/`flask` | 板上无 pip 生态 | 用标准库 `http.server`（本实现） |

---

## 8. 常驻还是按需？（**建议**，不是必须）

前面的实现用的是**按需拉起**（收请求才 fork 一次 `sd-cli`，跑完进程退出、显存归还）。
这是我们在"同机已有常驻 LLM 服务"这一场景下的选择；**换成常驻 `sd-server` 也能用**，
取决于你的场景。

### 两种形态的对照

| 维度 | 按需拉起（本文实现） | 常驻 `sd-server`（sd.cpp 自带） |
|---|---|---|
| 空闲显存占用 | **0** | 模型常驻，占池内 7–10 GB |
| 首张延迟 | 每次多 1–2 s **冷加载**（相对 57 s 出图约 2–4%；热载仅 0.4 s） | 无加载，但首次也要预热 |
| 同机其他服务重启 | 不受影响（池余量始终留着） | **可能抢不到内存起不来** |
| 开机启动顺序 | 无所谓（空闲不占） | 变成"先到先得"，需自行排序 |
| 实现复杂度 | 一个单文件服务（本文已给全） | 现成，配置即用 |
| 并发 | 单工作线程串行 + 队列（可查深度） | 自带并发（但一块 GPU 仍是串行瓶颈） |
| 现成生态 | 自己写的接口（OpenAI / A1111 兼容） | 自带 OpenAI + A1111 兼容端点 |

### 我们的建议

- **同机已有常驻 LLM 服务**（本平台的常见形态）⇒ 优先**按需拉起**：
  空闲零占用，不会因为扩散模型常驻而挤掉 LLM 的重启空间；
- **专职出图机 / 池只给扩散用** ⇒ **常驻 `sd-server` 更省事**，少一次加载、直接拿到现成端点；
- **两种都建议把"只回路径 + 异步 job"补上**（本文 §5 的两条）——如果调用方是 AI/Agent，
  这两条跟部署形态无关，是调用契约层面的必需品。

决定权在部署者：本节的目的是把取舍讲清楚，不是替你做选择。
---

[整理者注] 本文由真实部署过程整理；内部主机名、账号、凭据与业务标识已按公开分享规范移除或代称化
（如板端数据分区统一写作 `/brand_data`）。文中的命令与数据均来自实际运行记录。
