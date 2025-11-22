# Contributing to Netlist

This repository contains a Python-based netlist parsing and translation tool. It supports converting between Spectre, SPICE, and Xyce formats.

## Architecture Overview

The conversion process follows a standard compiler pipeline:

1.  **Lexing/Parsing**: Source files are read and parsed into a raw AST.
    *   `netlist/dialects/`: Contains dialect-specific parsers (Spectre, SPICE).
    *   `netlist/lex.py`: Token definitions.
2.  **Hierarchy Collection**: The raw statements are organized into a hierarchical structure (subcircuits, models, etc.).
    *   `netlist/parse.py`: `HierarchyCollector` class.
3.  **AST (Abstract Syntax Tree)**: The intermediate representation of the netlist.
    *   `netlist/data.py`: Dataclasses defining the AST nodes (`ModelDef`, `Instance`, `ParamDecl`, etc.).
4.  **Transformation (Optional)**: Passes that modify the AST (e.g., mapping parameters).
    *   `netlist/transform.py`: Prototype transformation passes.
5.  **Writing**: The AST is written out to the target format.
    *   `netlist/write/spice.py`: Writer implementation (currently focused on Xyce/SPICE output).

## Development Setup

1.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```

2.  Run tests:
    ```bash
    poetry run pytest
    ```

## Testing Strategy

The test suite is split into unit and integration tests:

*   **`tests/unit/`**: Focused tests for specific components.
    *   `test_parser.py`: Verifies that netlist strings are correctly parsed into AST nodes.
    *   `test_writer.py`: Verifies that AST nodes are correctly written to string output.
    *   `test_data.py`: Tests data structure behavior.
    *   `test_transform.py`: Tests AST transformation passes.
*   **`tests/integration/`**: End-to-end conversion tests.

### Writing Tests

*   Prefer **AST-based assertions** for parsers: assert that the output object equals the expected dataclass structure.
*   Prefer **Unit tests** over large integration tests for specific features (e.g., testing how a specific parameter is parsed).
*   Use **`dedent`** for multi-line string literals in tests to keep them readable.

## Code Style

*   We use `black` for formatting (not strictly enforced yet, but recommended).
*   Type hints are required for most function signatures.

