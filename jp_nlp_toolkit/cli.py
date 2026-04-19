"""CLI (typer)。"""
from __future__ import annotations

import typer

app = typer.Typer(help="jp-nlp-toolkit CLI")


@app.command()
def version():
    """バージョンを表示。"""
    from . import __version__
    typer.echo(f"jp-nlp-toolkit {__version__}")


@app.command()
def gui(host: str = "0.0.0.0", port: int = 8501):
    """Streamlit GUI を起動。"""
    import subprocess
    import sys
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", host, "--server.port", str(port),
    ])


if __name__ == "__main__":
    app()
