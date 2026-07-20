"""Lance l'agent de mémoire local et ouvre son interface dans le navigateur."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory_agent.server import main  # noqa: E402


def _selected_port(arguments: list[str]) -> int:
    for index, argument in enumerate(arguments):
        if argument == "--port" and index + 1 < len(arguments):
            try:
                return int(arguments[index + 1])
            except ValueError:
                return 8765
        if argument.startswith("--port="):
            try:
                return int(argument.split("=", 1)[1])
            except ValueError:
                return 8765
    return 8765


def _open_interface(port: int) -> None:
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    open_browser = "--no-browser" not in arguments
    arguments = [argument for argument in arguments if argument != "--no-browser"]
    if open_browser:
        threading.Thread(
            target=_open_interface,
            args=(_selected_port(arguments),),
            daemon=True,
        ).start()
    raise SystemExit(main(arguments))
