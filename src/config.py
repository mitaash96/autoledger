import os
import shutil

_HOME = os.path.expanduser("~")

concurrency_model = {"email": {"concurrency": "thread", "max_workers": 15}}

ollama = {"base_url": "http://localhost:11434"}

# Sandboxed opencode codegen backend. Paths are resolved per-machine; the sandbox
# root lives inside the autogen package so its assets are user-inspectable.
opencode = {
    "binary": shutil.which("opencode") or os.path.join(_HOME, ".opencode/bin/opencode"),
    "bwrap_binary": shutil.which("bwrap") or "/usr/bin/bwrap",
    "install_dir": os.path.join(_HOME, ".opencode"),
    "auth_dir": os.path.join(_HOME, ".local/share/opencode"),
    "sandbox_root": "src/autogen/sandbox",
    "venv_dir": ".venv",
    "default_model": "opencode/deepseek-v4-flash-free",
    "timeout_seconds": 1800,
    "max_ralph_rounds": 3,
    "env_passthrough": ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
}

io_dir = {"control": "data/control", "email": {"output": "data/pdf"}}

mail_query = """
has:attachment from:(icici OR estatement OR sbi OR hsbc OR hdfc OR credit_cards@icicibank.com OR emailstatements.cards@hdfcbank.net) -from:securities subject:(statement OR statements)
"""

banks = ["hsbc", "sbi", "icici", "hdfc"]

logging = {
    "level": "INFO",
    "log_file": "logs/pipeline.log",
    "process_log_file": "logs/process.log",
}
