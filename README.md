# DehazingEcho2025 Submission Example


> **Submission template for the DehazingEcho2025 Challenge Debug Phase**

This repository provides a Docker-based submission framework for participants. Follow these steps to package your dehazing model for evaluation.

## 🚀 Quick Start

### Step 1: Install Docker

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system.

### Step 2: Customize Your Model

#### Edit `inference.py`
Replace the placeholder with your model implementation:

#### Add Dependencies
- **Option A:** Add packages to `requirements.txt`
- **Option B:** Place `.whl` files in `resources/python_packages/`

#### Add Model Files
Place your trained models in `model/`

### Step 3: Build and Submit

```bash
# To run the container locally, you can call the following bash script:

  ./do_test_run.sh

# This will start the inference and reads from ./test/input and writes to ./test/output

# To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh
```

**Good luck! 🚀**
