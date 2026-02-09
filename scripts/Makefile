.PHONY: setup download-fonts prepare-data train-style-encoder train-kerning-net \
       train-glyph-diffusion evaluate convert-all test lint clean help

PYTHON ?= python3
PIP ?= pip3
DATA_DIR ?= data/fonts
EXTRACTED_DIR ?= data/extracted
CHECKPOINT_DIR ?= checkpoints
COREML_DIR ?= ../Typogenesis/Resources/Models
DEVICE ?= auto

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and dev tools
	$(PIP) install -e ".[dev]"

download-fonts: ## Download Google Fonts (Latin subset)
	$(PYTHON) -m data.download_fonts --output-dir $(DATA_DIR) --limit 1500

prepare-data: ## Extract glyphs and build dataset manifests
	$(PYTHON) -m data.extract_glyphs --font-dir $(DATA_DIR) --output-dir $(EXTRACTED_DIR)
	$(PYTHON) -m data.prepare_datasets --data-dir $(EXTRACTED_DIR)

train-style-encoder: ## Train StyleEncoder (run first - other models depend on this)
	$(PYTHON) -m models.style_encoder.train \
		--data_dir $(EXTRACTED_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR)/style_encoder \
		--epochs 100 \
		--batch_size 256

train-kerning-net: ## Train KerningNet
	$(PYTHON) -m models.kerning_net.train \
		--data_dir $(EXTRACTED_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR)/kerning_net \
		--epochs 50 \
		--batch_size 128

train-glyph-diffusion: ## Train GlyphDiffusion (requires trained StyleEncoder)
	$(PYTHON) -m models.glyph_diffusion.train \
		--config default \
		--data_dir $(EXTRACTED_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR)/glyph_diffusion \
		--epochs 200

evaluate: ## Run evaluation metrics on all trained models
	$(PYTHON) -m evaluation.metrics \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--data-dir $(EXTRACTED_DIR) \
		--output-dir outputs/evaluation

convert-all: ## Convert all trained models to CoreML
	$(PYTHON) -m convert.convert_style_encoder \
		--checkpoint $(CHECKPOINT_DIR)/style_encoder/best.pt \
		--output $(COREML_DIR)/StyleEncoder.mlpackage
	$(PYTHON) -m convert.convert_kerning_net \
		--checkpoint $(CHECKPOINT_DIR)/kerning_net/best.pt \
		--output $(COREML_DIR)/KerningNet.mlpackage
	$(PYTHON) -m convert.convert_glyph_diffusion \
		--checkpoint $(CHECKPOINT_DIR)/glyph_diffusion/best.pt \
		--output $(COREML_DIR)/GlyphDiffusion.mlpackage

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-fast: ## Run tests excluding slow/gpu/data markers
	$(PYTHON) -m pytest tests/ -v -m "not slow and not gpu and not data"

lint: ## Run linter
	$(PYTHON) -m ruff check .

lint-fix: ## Run linter with auto-fix
	$(PYTHON) -m ruff check . --fix

clean: ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
