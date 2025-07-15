# This prevents 'make' from getting confused if a file named 'init' happens to exist.
.PHONY: init help

# Default target when 'make' is run without arguments
all: help

# Help target to explain available commands
help:
	@echo "Available commands:"
	@echo "  make init   - Initializes the project environment (runs init.sh)"
	@echo "  make help   - Show this help message"

# Init target: Runs the init.sh script
init:
	@echo "--- Running project initialization script (init.sh) ---"
	# Execute the init.sh script.
	bash init.sh
	@echo "--- Initialization complete ---"