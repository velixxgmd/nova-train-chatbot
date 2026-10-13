import psutil

TARGET = 16176
for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
    if p.info["pid"] == TARGET and p.info["name"] == "python.exe":
        rss = p.info["memory_info"].rss / 1e6
        cmd = " ".join(p.info["cmdline"] or [])[:90]
        print(f"train PID {TARGET} RSS: {rss:.0f} MB | {cmd}")
vm = psutil.virtual_memory()
print(f"system: total {vm.total/1e9:.1f} GB | available {vm.available/1e9:.1f} GB | used {vm.percent:.0f}%")
