"""Enable `python -m agent_cli` for development."""
from agent_cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
