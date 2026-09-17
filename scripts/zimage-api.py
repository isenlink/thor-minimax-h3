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
