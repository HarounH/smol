import sys
import logging

# Define the colors for different log levels
LOG_COLORS = {
    'WARNING': '\033[93m',  # Pastel yellow
    'ERROR': '\033[91m',    # Pastel red
    'DEBUG': '\033[91m',    # Pastel green
    'INFO': '\033[92m',
}

# Custom log format with colors
LOG_FORMAT = (
    '[%(levelname).1s:%(asctime)s][%(name)s:%(lineno)d] %(message)s'
)

logger_initialized = False

# Create a custom formatter that adds colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, '\033[0m')
        log_message = super().format(record)
        return f"{log_color}{log_message}\033[0m"

# Create and configure the logger
def setup_logger():
    global logger_initialized
    if logger_initialized:
        return

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger() # root logger
    logger.handlers.clear()

    # Create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a custom formatter and set it for the handler
    formatter = ColoredFormatter(LOG_FORMAT)
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

# Call the setup_logger function to configure the logger
setup_logger()
