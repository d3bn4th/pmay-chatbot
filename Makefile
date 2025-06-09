SHELL :=/bin/bash

.PHONY: clean check setup run install-models help
.DEFAULT_GOAL=help
VENV_DIR = .venv
PYTHON_VERSION = python3.11

check: # Run linting checks
	@echo "🔍 Running linting checks..."
	@ruff check .
	@echo "✅ Linting checks complete!"

fix: # Fix auto-fixable linting issues
	@echo "🔧 Fixing linting issues..."
	@ruff check . --fix
	@echo "✅ Linting fixes complete!"

clean: # Clean temporary files and caches
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ .pytest_cache
	@find . -name '*.pyc' -exec rm -r {} +
	@find . -name '__pycache__' -exec rm -r {} +
	@rm -rf build dist
	@find . -name '*.egg-info' -type d -exec rm -r {} +
	@rm -rf .ruff_cache
	@echo "✅ Cleanup complete!"

run: # Run the PMAY Chatbot application
	@echo "🚀 Starting PMAY Chatbot..."
	@streamlit run app.py

setup: # Initial project setup
	@echo "🔧 Setting up PMAY Chatbot project..."
	@echo "Creating virtual env at: $(VENV_DIR)"
	@$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	@source $(VENV_DIR)/bin/activate && pip install -r requirements/requirements-dev.txt && pip install -r requirements/requirements.txt
	@echo -e "\n✅ Setup complete!\n🎉 Run the following commands to get started:\n\n ➡️ source $(VENV_DIR)/bin/activate\n ➡️ make install-models\n ➡️ make run\n"

install-models: # Install required Ollama models
	@echo "📥 Installing required Ollama models..."
	@ollama pull nomic-embed-text:latest
	@ollama pull llama3.2:1b
	@echo "✅ Models installed successfully!"

check-ollama: # Check if Ollama is running
	@echo "🔍 Checking Ollama status..."
	@curl -s http://localhost:11434/api/tags > /dev/null || (echo "❌ Ollama is not running. Please start Ollama first." && exit 1)
	@echo "✅ Ollama is running!"

help: # Show this help
	@echo "🤖 PMAY Chatbot - Available Commands:"
	@echo ""
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
