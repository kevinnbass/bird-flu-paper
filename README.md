# Bird Flu Analysis Processor

## Overview
This project processes articles about bird flu transmission through a series of analysis phases using the DeepSeek API.

## Project Structure
```
bird_flu_analysis/
│
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── prompts.yaml
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── processor.py
│   └── validators.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_processor.py
│   └── test_validators.py
│
├── logs/
├── .env
├── .env.example
├── requirements.txt
└── main.py
```

## Setup
1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```
5. Configure settings in `config/config.yaml`

## Usage
Run the main script:
```bash
python main.py
```

## Testing
Run tests with:
```bash
pytest
```

## Development
- Format code: `black .`
- Sort imports: `isort .`
- Check types: `mypy .`
- Lint code: `flake8`
