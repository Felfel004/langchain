class Colors:
    PURPLE = "\033[95m"
    YELLOW = "\033[93m"
    DARKCYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_info(message, color=""):
    print(f"{color}[INFO] {message}{Colors.RESET}")


def log_success(message):
    print(f"\033[92m[SUCCESS] {message}{Colors.RESET}")


def log_warning(message):
    print(f"\033[93m[WARNING] {message}{Colors.RESET}")


def log_error(message):
    print(f"\033[91m[ERROR] {message}{Colors.RESET}")


def log_header(message):
    print()
    print("=" * 60)
    print(message)
    print("=" * 60)