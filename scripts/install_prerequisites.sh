#!/bin/bash

set -euo pipefail 

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Change to script directory and load environment from parent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source .env
source "$SCRIPT_DIR/utils.sh"
cd "$ROOT_PATH"

# Directory paths
UTILS_DIR="$ROOT_PATH/utils"
VENV_PATH="$ROOT_PATH/.ws_pip"

# Shell configuration
CURRENT_SHELL=$(basename "$SHELL")
case "$CURRENT_SHELL" in
"zsh" | "zsh5" | "/bin/zsh" | "/usr/bin/zsh")
    SHELL_CONFIG="$HOME/.zshrc"
    SHELL_TYPE="zsh"
    ;;
"bash" | "bash5" | "/bin/bash" | "/usr/bin/bash")
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_TYPE="bash"
    ;;
*)
    echo "Warning: Unsupported shell '$CURRENT_SHELL' detected. Defaulting to bash configuration."
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_TYPE="bash"
    ;;
esac

# Package lists
DEV_PACKAGES=" \
    git curl wget \
    python3 python3-pip \
    build-essential cmake \
    unzip pkg-config \
    systemtap-sdt-dev \
    "

PYENV_PACKAGES=" \
    libssl-dev zlib1g-dev libbz2-dev ncurses-dev \
    libffi-dev libreadline-dev sqlite3 libsqlite3-dev \
    tk-dev bzip2 libbz2-dev lzma liblzma-dev \
    llvm libncursesw5-dev xz-utils \
    libxml2-dev libxmlsec1-dev \
    "

# Python version configuration
PYTHON_VERSION="3.10.16"
HERMETIC_PYTHON_VERSION="3.10"

# Clang version configuration
CLANG_VERSION="18"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Check if a package is installed
is_package_installed() {
    dpkg -l | grep -q "^ii  $1 "
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Add complete shell configuration block atomically
add_shell_config_block() {
    local config_file="$1"
    local block_content="$2"
    local block_marker="$3"

    # Check if block already exists
    if grep -q "$block_marker" "$config_file" 2>/dev/null; then
        log "Shell config block with marker '$block_marker' already exists in $config_file, skipping..."
        return
    fi

    log "Adding shell config block to $config_file..."
    echo "" >>"$config_file"
    echo "$block_content" >>"$config_file"
}

# Get missing packages from a package list
get_missing_packages() {
    local package_list="$1"
    local missing_packages=""

    for pkg in $package_list; do
        if ! is_package_installed "$pkg"; then
            missing_packages="$missing_packages $pkg"
        fi
    done

    echo "$missing_packages"
}

# =============================================================================
# INSTALLATION FUNCTIONS
# =============================================================================

setup_workspace() {
    log "Setting up workspace"
    log "Working in ROOT_PATH: $ROOT_PATH"
    log "Detected shell: $CURRENT_SHELL"
    log "Using config file: $SHELL_CONFIG (Shell type: $SHELL_TYPE)"

    # Ensure utils directory exists
    if [ ! -d "$UTILS_DIR" ]; then
        mkdir -p "$UTILS_DIR"
        log "Created utils directory: $UTILS_DIR"
    fi
}

install_dev_prerequisites() {
    log "Checking development prerequisites..."
    local missing_packages
    missing_packages=$(get_missing_packages "$DEV_PACKAGES")

    if [ -n "$missing_packages" ]; then
        log "Installing missing packages:$missing_packages"
        sudo apt update
        sudo apt install -y $missing_packages
    else
        log "All development prerequisites are already installed."
    fi
}

install_clang() {
    log "Checking clang version $CLANG_VERSION..."
    if command_exists "clang-$CLANG_VERSION"; then
        log "Clang $CLANG_VERSION is already installed."
        return
    fi

    log "Installing clang version $CLANG_VERSION..."
    cd "$UTILS_DIR"

    if [ ! -f llvm.sh ]; then
        log "Downloading llvm.sh to utils directory..."
        wget -O llvm.sh https://apt.llvm.org/llvm.sh
        chmod +x llvm.sh
    fi

    sudo ./llvm.sh "$CLANG_VERSION"
    cd "$ROOT_PATH"
}

install_bazel() {
    log "Checking bazelisk and bazel..."
    if command_exists "bazel" && command_exists "bazelisk"; then
        log "Bazelisk and bazel are already installed."
        return
    fi

    log "Installing bazelisk and bazel..."

    # Detect architecture
    local arch
    arch=$(uname -m)
    local bazelisk_arch
    case "$arch" in
    "x86_64")
        bazelisk_arch="linux-amd64"
        ;;
    "aarch64" | "arm64")
        bazelisk_arch="linux-arm64"
        ;;
    *)
        warn "Unsupported architecture $arch. Defaulting to linux-amd64"
        bazelisk_arch="linux-amd64"
        ;;
    esac

    log "Downloading bazelisk for architecture: $bazelisk_arch"
    cd "$UTILS_DIR"

    if [ ! -f bazelisk ]; then
        curl -L "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-$bazelisk_arch" -o bazelisk
        chmod +x bazelisk
    fi

    sudo cp bazelisk /usr/local/bin/bazelisk
    sudo ln -sf /usr/local/bin/bazelisk /usr/bin/bazel
    cd "$ROOT_PATH"
}

install_pyenv_prerequisites() {
    log "Checking pyenv prerequisites..."
    local missing_packages
    missing_packages=$(get_missing_packages "$PYENV_PACKAGES")

    if [ -n "$missing_packages" ]; then
        log "Installing missing pyenv prerequisites:$missing_packages"
        sudo apt install -y --no-install-recommends $missing_packages
    else
        log "All pyenv prerequisites are already installed."
    fi
}

install_pyenv() {
    log "Checking pyenv installation..."
    if command_exists "pyenv"; then
        log "Pyenv is already installed."
        return
    fi

    log "Installing pyenv..."
    cd "$UTILS_DIR"

    if [ ! -f pyenv-installer.sh ]; then
        log "Downloading pyenv installer to utils directory..."
        curl -fsSL https://pyenv.run -o pyenv-installer.sh
        chmod +x pyenv-installer.sh
    fi

    bash pyenv-installer.sh
    cd "$ROOT_PATH"
}

install_python() {
    log "Checking Python $PYTHON_VERSION installation..."

    # Setup pyenv environment for the current script run
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command_exists pyenv; then
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    else
        warn "pyenv command not found. Python installation might fail."
    fi

    if pyenv versions | grep -q "$PYTHON_VERSION"; then
        log "Python $PYTHON_VERSION is already installed via pyenv."
    else
        log "Installing Python $PYTHON_VERSION via pyenv..."
        pyenv install "$PYTHON_VERSION"
    fi

    pyenv global "$PYTHON_VERSION"
}

configure_python_venv() {
    log "Checking Python virtual environment..."

    if [ -d "$VENV_PATH" ]; then
        log "Python virtual environment already exists."
    else
        log "Creating Python virtual environment..."
        cd "$ROOT_PATH"
        python3 -m venv .ws_pip
        cd "$ROOT_PATH"
    fi
}

configure_shell() {
    log "Configuring shell for pyenv, virtual environment, and Bazel..."

    local block_marker="# Auto-generated settings by install_prerequisites.sh"
    local config_block="
    # ==================================================================
    $block_marker

    # Pyenv configuration for Python version management
    export PYENV_ROOT=\"\$HOME/.pyenv\"
    export PATH=\"\$PYENV_ROOT/bin:\$PATH\"
    if command -v pyenv 1>/dev/null 2>&1; then
    eval \"\$(pyenv init --path)\"
    eval \"\$(pyenv init -)\"
    fi

    # Python virtual environment auto-activation
    if [ -f \"$VENV_PATH/bin/activate\" ]; then
        source \"$VENV_PATH/bin/activate\"
    fi

    # Bazel hermetic Python version specification
    export HERMETIC_PYTHON_VERSION=$HERMETIC_PYTHON_VERSION
    # ==================================================================
    "

    add_shell_config_block "$SHELL_CONFIG" "$config_block" "$block_marker"
}




# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_info "Starting prerequisites installation script..."
    setup_workspace
    install_dev_prerequisites
    install_clang
    install_bazel
    install_pyenv_prerequisites
    install_pyenv
    install_python
    configure_python_venv
    configure_shell

    log_info "Installation completed!"
    log_info "Please restart your shell or run the following to apply all changes:"
    log_info "  source $SHELL_CONFIG"

}

# Execute main function
main "$@"

# (optional) install zsh
##################
# # Step 1: Install zsh, oh-my-zsh, powerlevel10k theme
# sudo apt install zsh -y
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# # Step 2: Replace ZSH_THEME in ~/.zshrc
# sed -i 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc

# # zsh-syntax-highlighting
# git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# # zsh-autosuggestions
# git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# # update .zshrc
# # 7. plugins 라인 수정 (기존에 있으면 교체)
# if grep -q '^plugins=' ~/.zshrc; then
#   sed -i 's/^plugins=.*/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/' ~/.zshrc
# else
#   echo 'plugins=(git zsh-syntax-highlighting zsh-autosuggestions)' >> ~/.zshrc
# fi

# # Step 3: Add p10k auto-config logic if not exists
# if ! grep -q 'p10k configure' ~/.zshrc; then
#   echo -e '\n[[ ! -f ~/.p10k.zsh ]] && p10k configure' >> ~/.zshrc
#   echo '[[ -f ~/.p10k.zsh ]] && source ~/.p10k.zsh' >> ~/.zshrc
# fi

# # Step 4: Set zsh as default shell (optional)
# chsh -s $(which zsh)

# # update .zshrc to support pyevn
# # echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
# # echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
# # echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
# # source ~/.zshrc
##################

# sudo apt install libopencv-dev -y
# wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip
# wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip
# unzip opencv.zip
# unzip opencv_contrib.zip
# mkdir opencv_build
# cd opencv_build
# cmake -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/usr/src \
#     -DWITH_OPENCL=OFF \
#     -DWITH_OPENCL_SVM=OFF \
#     -DWITH_OPENCL_D3D11_NV=OFF \
#     -DWITH_OPENGL=OFF \
#     -DBUILD_opencv_world=ON \
#     -DBUILD_TESTS=OFF \
#     -DBUILD_PERF_TESTS=OFF \
#     -DWITH_TBB=OFF \
#     -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.11.0/modules/ \
#     ../opencv-4.11.0/

# make -j4
