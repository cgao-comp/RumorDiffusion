#!/bin/bash

# =============================================================================
# Python Project Cleanup Script
# 
# This script performs comprehensive cleanup of Python project artifacts
# including cache files, temporary files, build artifacts, log files,
# machine learning artifacts, and visualization files.
# =============================================================================

set -euo pipefail

# Color codes for output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Global variables
VERBOSE=false
DRY_RUN=false
CLEANUP_COUNT=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to safely remove directories by pattern
safe_remove_dir() {
    local pattern="$1"
    local description="$2"
    
    [[ -z "$pattern" ]] && log_error "Pattern is empty for: $description" && return 1
    
    log_info "Cleaning $description..."
    
    local count=$(find . -type d -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        find . -type d -name "$pattern" 2>/dev/null | while read -r dir; do
            echo "  Would remove: $dir"
        done
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
        return 0
    fi
    
    if find . -type d -name "$pattern" -exec rm -rf {} + 2>/dev/null; then
        log_success "Removed $count $description"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Function to safely remove files by pattern
safe_remove_files() {
    local pattern="$1"
    local description="$2"
    
    [[ -z "$pattern" ]] && log_error "Pattern is empty for: $description" && return 1
    
    log_info "Cleaning $description..."
    
    local count=$(find . -type f -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        find . -type f -name "$pattern" 2>/dev/null | head -10 | while read -r file; do
            echo "  Would remove: $file"
        done
        [[ $count -gt 10 ]] && echo "  ... and $((count - 10)) more files"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
        return 0
    fi
    
    if find . -type f -name "$pattern" -delete 2>/dev/null; then
        log_success "Removed $count $description"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Function to safely remove specified path (file or directory)
safe_remove_path() {
    local path="$1"
    local description="$2"
    
    [[ -z "$path" ]] && log_error "Path is empty for: $description" && return 1
    
    log_info "Cleaning $description..."
    
    if [[ ! -e "$path" ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would remove: $path"
        CLEANUP_COUNT=$((CLEANUP_COUNT + 1))
        return 0
    fi
    
    if rm -rf "$path" 2>/dev/null; then
        log_success "Removed $description: $path"
        CLEANUP_COUNT=$((CLEANUP_COUNT + 1))
    else
        log_warning "Could not remove $description: $path (may be in use)"
    fi
}

# Function to remove directories matching pattern in specific location
safe_remove_dir_in_path() {
    local base_path="$1"
    local pattern="$2"
    local description="$3"
    
    [[ -z "$base_path" ]] || [[ -z "$pattern" ]] && return 1
    
    log_info "Cleaning $description in $base_path..."
    
    if [[ ! -d "$base_path" ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    local count=$(find "$base_path" -type d -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        find "$base_path" -type d -name "$pattern" 2>/dev/null | while read -r dir; do
            echo "  Would remove: $dir"
        done
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
        return 0
    fi
    
    if find "$base_path" -type d -name "$pattern" -exec rm -rf {} + 2>/dev/null; then
        log_success "Removed $count $description"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Function to remove files matching pattern in specific location
safe_remove_files_in_path() {
    local base_path="$1"
    local pattern="$2"
    local description="$3"
    
    [[ -z "$base_path" ]] || [[ -z "$pattern" ]] && return 1
    
    log_info "Cleaning $description in $base_path..."
    
    if [[ ! -d "$base_path" ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    local count=$(find "$base_path" -type f -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        find "$base_path" -type f -name "$pattern" 2>/dev/null | head -10 | while read -r file; do
            echo "  Would remove: $file"
        done
        [[ $count -gt 10 ]] && echo "  ... and $((count - 10)) more files"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
        return 0
    fi
    
    if find "$base_path" -type f -name "$pattern" -delete 2>/dev/null; then
        log_success "Removed $count $description"
        CLEANUP_COUNT=$((CLEANUP_COUNT + count))
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Clean Python cache and compiled files
clean_python_cache() {
    log_info "=== Cleaning Python Cache ==="
    
    safe_remove_dir "__pycache__" "Python cache directories"
    safe_remove_dir ".pytest_cache" "pytest cache directories"
    safe_remove_dir ".cache" "pytest cache directories"
    safe_remove_dir ".hypothesis" "hypothesis cache directories"
    safe_remove_dir ".mypy_cache" "mypy cache directories"
    safe_remove_dir ".coverage" "coverage cache directories"
    safe_remove_dir ".ipynb_checkpoints" "Jupyter notebook checkpoints"
    
    safe_remove_files "*.pyc" "Python compiled files"
    safe_remove_files "*.pyo" "Python optimized files"
    safe_remove_files "*.pyd" "Python extension modules"
}

# Clean log files
clean_logs() {
    log_info "=== Cleaning Log Files ==="
    
    safe_remove_files "*.log" "Log files"
    safe_remove_files "*.out" "Output log files"
    safe_remove_files "*.err" "Error log files"
    safe_remove_files "slurm-*.out" "Slurm output files"
    safe_remove_files "slurm-*.err" "Slurm error files"
    
    safe_remove_path "Output/logs" "Log directory"
    safe_remove_files_in_path "Output/logs" "*.log" "Benchmark log files"
    safe_remove_files_in_path "Output/logs" "*_single_job_id.txt" "Single job ID files"
    safe_remove_files_in_path "Output/logs" "*_job_ids.csv" "Job ID CSV files"
}

# Clean machine learning artifacts
clean_ml_artifacts() {
    log_info "=== Cleaning Machine Learning Artifacts ==="
    
    safe_remove_path "Output/ckpt" "Model checkpoints directory"
    safe_remove_dir "ckpt" "PyTorch model checkpoints"
    
    safe_remove_path "runs" "TensorBoard runs directory"
    safe_remove_path "tensorboard_logs" "TensorBoard logs directory"
    safe_remove_path "wandb" "Weights & Biases cache directory"
    
    safe_remove_path "Output/results" "Results directory"
    safe_remove_files_in_path "Output/results" "*.csv" "Result CSV files"
    safe_remove_files_in_path "Output/results/csv" "*.csv" "Experiment result CSV files"
    
    safe_remove_path "Output/slurm_jobs" "Slurm job scripts directory"
}

# Clean cache files
clean_cache() {
    log_info "=== Cleaning Cache Files ==="
    
    safe_remove_path "cache" "Project cache directory"
    
    safe_remove_dir_in_path "Input" "cache" "Dataset cache directories"
    safe_remove_dir_in_path "Input/ABeta" "cache" "ABeta cache directory"
    safe_remove_dir_in_path "Input/MRI" "cache" "MRI cache directory"
    safe_remove_dir_in_path "Input/NaK" "cache" "NaK cache directory"
    safe_remove_dir_in_path "Input/SD" "cache" "SD cache directory"
}

# Clean visualization files
clean_visualizations() {
    log_info "=== Cleaning Visualization Files ==="
    
    safe_remove_files_in_path "Output/visualizations" "*.npz" "Visualization NPZ files"
    safe_remove_path "Output/visualizations/individual" "Individual model visualizations"
    safe_remove_files_in_path "Output/visualizations" "comparison_registry.json" "Comparison registry"
}

# Clean build and distribution artifacts
clean_build_artifacts() {
    log_info "=== Cleaning Build Artifacts ==="
    
    safe_remove_dir "build" "build directories"
    safe_remove_dir "dist" "distribution directories"
    safe_remove_dir "*.egg-info" "egg-info directories"
}

# Clean IDE and editor files
clean_ide_files() {
    log_info "=== Cleaning IDE and Editor Files ==="
    
    safe_remove_dir ".vscode" "VSCode settings"
    safe_remove_dir ".idea" "PyCharm/IntelliJ settings"
    
    safe_remove_files "*.swp" "Vim swap files"
    safe_remove_files "*.swo" "Vim swap files"
    safe_remove_files "*~" "Backup files"
}

# Clean temporary files
clean_temp_files() {
    log_info "=== Cleaning Temporary Files ==="
    
    safe_remove_files "*.tmp" "Temporary files"
    safe_remove_files "*.temp" "Temporary files"
    safe_remove_files ".DS_Store" "macOS system files"
}

# Main cleanup function
main() {
    log_info "Starting Python project cleanup..."
    log_info "Working directory: $(pwd)"
    
    if git rev-parse --git-dir > /dev/null 2>&1; then
        log_info "Git repository detected"
    else
        log_warning "Not in a git repository - be careful with cleanup"
    fi
    
    if [[ -d "venv" ]] || [[ -d ".venv" ]] || [[ -d "env" ]]; then
        log_warning "Virtual environment detected - skipping venv cleanup for safety"
    fi
    
    clean_python_cache
    clean_logs
    clean_ml_artifacts
    clean_cache
    clean_visualizations
    clean_build_artifacts
    clean_ide_files
    clean_temp_files
    
    log_success "Cleanup completed successfully!"
    log_info "Total items cleaned: $CLEANUP_COUNT"
    
    if command -v du >/dev/null 2>&1; then
        log_info "Current directory size: $(du -sh . | cut -f1)"
    fi
}

# Dry run preview function
dry_run_preview() {
    log_warning "DRY RUN MODE - No files will be actually deleted"
    log_info "Would clean the following items:"
    echo ""
    
    log_info "=== Python Cache ==="
    find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".coverage" -o -name ".ipynb_checkpoints" \) 2>/dev/null | head -20
    find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) 2>/dev/null | head -20
    
    log_info "=== Log Files ==="
    find . -type f \( -name "*.log" -o -name "*.out" -o -name "*.err" -o -name "slurm-*.out" -o -name "slurm-*.err" \) 2>/dev/null | head -20
    [[ -d "Output/logs" ]] && find Output/logs -type f 2>/dev/null | head -10
    
    log_info "=== Machine Learning Artifacts ==="
    [[ -d "Output/ckpt" ]] && find Output/ckpt -type f 2>/dev/null | head -10
    [[ -d "runs" ]] && echo "runs/"
    [[ -d "tensorboard_logs" ]] && echo "tensorboard_logs/"
    [[ -d "wandb" ]] && echo "wandb/"
    [[ -d "Output/results" ]] && find Output/results -name "*.csv" 2>/dev/null | head -10
    [[ -d "Output/slurm_jobs" ]] && find Output/slurm_jobs -type f 2>/dev/null | head -10
    
    log_info "=== Cache Files ==="
    [[ -d "cache" ]] && echo "cache/"
    find Input -type d -name "cache" 2>/dev/null
    
    log_info "=== Visualization Files ==="
    [[ -d "Output/visualizations" ]] && find Output/visualizations -name "*.npz" 2>/dev/null | head -10
    [[ -d "Output/visualizations/individual" ]] && echo "Output/visualizations/individual/"
    
    log_info "=== Build Artifacts ==="
    find . -type d \( -name "build" -o -name "dist" -o -name "*.egg-info" \) 2>/dev/null | head -10
    
    log_info "=== IDE and Editor Files ==="
    find . -type d \( -name ".vscode" -o -name ".idea" \) 2>/dev/null
    find . -type f \( -name "*.swp" -o -name "*.swo" -o -name "*~" \) 2>/dev/null | head -10
    
    log_info "=== Temporary Files ==="
    find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name ".DS_Store" \) 2>/dev/null | head -10
    
    echo ""
    log_info "Use '$0' (without --dry-run) to actually perform cleanup"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Python Project Cleanup Script

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -d, --dry-run   Show what would be cleaned without actually cleaning

EXAMPLES:
    $0              # Run normal cleanup
    $0 --dry-run    # Show what would be cleaned
    $0 --verbose    # Run with verbose output

This script cleans:
- Python cache directories (__pycache__, .pytest_cache, etc.)
- Compiled Python files (*.pyc, *.pyo)
- Log files (*.log, *.out, *.err, slurm-*.out)
- Machine learning artifacts (checkpoints, TensorBoard logs, W&B cache)
- Build artifacts (build/, dist/, *.egg-info)
- IDE settings (.vscode, .idea)
- Temporary and backup files
- Cache files (cache/, Input/*/cache/)
- Visualization files (Output/visualizations/*.npz, individual/)
- Results files (Output/results/*.csv)

WARNING: This script will permanently delete files. Use --dry-run first.
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set verbose mode
[[ "$VERBOSE" == true ]] && set -x

# Execute dry run or main cleanup
if [[ "$DRY_RUN" == true ]]; then
    dry_run_preview
else
    main "$@"
fi
