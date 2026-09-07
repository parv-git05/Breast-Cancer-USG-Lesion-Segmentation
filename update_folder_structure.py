import os

# Files to output
OUTPUT_FILE = "project_structure.txt"

# Directory & file ignore patterns
IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    ".idea",
    ".gemini",
    "node_modules",
}

IGNORE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    OUTPUT_FILE,
}

MAX_FILES_PER_DIR = 12

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def generate_tree(start_dir):
    lines = []
    lines.append(f"Project Folder Structure: {os.path.basename(os.path.abspath(start_dir))}")
    lines.append("=" * 60)
    lines.append("")

    def _build_tree(dir_path, prefix=""):
        try:
            entries = os.listdir(dir_path)
        except PermissionError:
            return

        dirs = []
        files = []

        for entry in sorted(entries):
            full_path = os.path.join(dir_path, entry)
            if entry in IGNORE_DIRS or entry in IGNORE_FILES:
                continue
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                files.append(entry)

        all_items = dirs + files
        total_items = len(all_items)

        # Process dirs first, then files
        for i, d in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and (len(files) == 0)
            connector = "└── " if is_last_dir else "├── "
            lines.append(f"{prefix}{connector}{d}/")
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            _build_tree(os.path.join(dir_path, d), new_prefix)

        # Process files with truncating for huge datasets
        num_files = len(files)
        show_files = files if num_files <= MAX_FILES_PER_DIR else files[:MAX_FILES_PER_DIR]

        for i, f in enumerate(show_files):
            is_last = (i == num_files - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{f}")

        if num_files > MAX_FILES_PER_DIR:
            lines.append(f"{prefix}└── ... ({num_files - MAX_FILES_PER_DIR} additional files omitted for brevity)")

    _build_tree(start_dir)
    return "\n".join(lines)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    tree_text = generate_tree(root_dir)
    output_path = os.path.join(root_dir, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree_text)
    print(f"Updated {OUTPUT_FILE} successfully.")

if __name__ == "__main__":
    main()
