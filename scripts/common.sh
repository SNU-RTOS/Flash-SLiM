# ──────────────────────────────────────────────────────────────────────────────

create_symlink_or_fail() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [ ! -e "$src" ]; then
    echo "❌ Target not found: $src"
    exit 1
  fi

  echo "→ Making symlink: $label"
  ln -sf "$src" "$dst"
}