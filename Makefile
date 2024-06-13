## ------------------------------------------------------------
## CardioSigKit is a package that provides tools for analysing
## cardio bio signals
## ------------------------------------------------------------
## This file is intended to build or install the package.
## ------------------------------------------------------------

CPU = 30
BUILD_FOLDER = ./build
.PHONY: help install setup clean build build-system build-jupyter

help: banner ## Show this help.
	@echo "Usage: make [target] ..."
	@echo ""
	@echo "Miscellaneous:"
	@awk '/^help:.*?##/ {split($$0, a, ":.*?##"); printf "  %-20s %s\n", a[1], a[2]}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Build:"
	@awk '/^build[^ ]*:.*?##/ {split($$0, a, ":.*?##"); printf "  %-20s %s\n", a[1], a[2]}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Cleaning:"
	@awk '/^clean[^ ]*:.*?##/ {split($$0, a, ":.*?##"); printf "  %-20s %s\n", a[1], a[2]}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Written by Ahmad Fall, 10/11/2023"
	@echo "Please report any bug or error to the author <ahmad.fall@ird.fr>."

install-user: clean setup ## Build and install in user's home folder.
	pip install ./ --global-option="build_ext" --global-option="-j$(CPU)" --user

build: clean setup ## Build only the package in folder specified with variable 'BUILD_FOLDER'
	python setup.py build -j $(CPU) --build-lib $(BUILD_FOLDER)

setup: ## Install python required packages
	pip install -r requirements.txt

clean: banner ## Clean all build, cache, compiled and temporary files
	find ./ -name '*.c' -type f -delete
	find ./ -name '*.cpp' -type f -delete
	find ./ | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf build cython_debug

clean-uninstall: ## Uninstall the library
	pip uninstall ecgdatakit --yes

banner:
	@echo "-------------------------------------------------------------"
	@echo "CardioSigKit is a package that provides tools for analysing"
	@echo "cardio bio signals"
	@echo "------------------------------------------------------------"
	@echo "This file is intended to build or install the package."
	@echo "------------------------------------------------------------"