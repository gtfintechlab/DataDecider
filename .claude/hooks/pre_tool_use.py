#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
import json
import sys
import re

def modify_command(command):
    """Modify commands to enforce uv usage."""
    # Replace 'python' or 'python3' with 'uv run python'
    command = re.sub(r'\bpython3?\b', 'uv run python', command)
    # Replace 'pip install' with 'uv add'
    command = re.sub(r'\bpip\s+install\b', 'uv add', command)
    # Replace 'pip uninstall' with 'uv remove'
    command = re.sub(r'\bpip\s+uninstall\b', 'uv remove', command)
    return command

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        command = tool_input.get('command', '')

        if tool_name == 'Bash' and command:
            modified_command = modify_command(command)
            if modified_command != command:
                # Output feedback to Claude
                print(f"BLOCKED: Replaced command with uv equivalent: {modified_command}", file=sys.stderr)
                print(json.dumps({"decision": "block", "reason": f"Use {modified_command} instead"}), file=sys.stdout)
                sys.exit(2)  # Exit code 2 blocks the original command and provides feedback

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()