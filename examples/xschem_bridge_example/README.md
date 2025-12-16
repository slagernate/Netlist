# Xschem Bridge File Example

This example demonstrates generating xschem symbols and dialect-specific bridge files using the Model Indirection pattern.

## Overview

The example creates a simple IR with 6 primitives:
- `nmos` - NMOS transistor
- `pmos` - PMOS transistor  
- `npn_bjt` - NPN bipolar transistor
- `resistor` - Resistor
- `capacitor` - Capacitor
- `diode` - Diode

Each primitive has its own model definition.

## Running the Example

```bash
cd examples/xschem_bridge_example
PYTHONPATH=../.. python3 xschem_bridge_example.py
```

This will generate:
- 6 `.sym` symbol files (one per primitive, using IR names)
- `bridge_xyce.spice` - Bridge file for Xyce simulator
- `bridge_ngspice.spice` - Bridge file for Ngspice simulator

## Output Files

### Symbol Files

The symbol files are generated in `output/`:
- `nmos.sym` - NMOS symbol
- `pmos.sym` - PMOS symbol
- `npn_bjt.sym` - BJT symbol
- `resistor.sym` - Resistor symbol
- `capacitor.sym` - Capacitor symbol
- `diode.sym` - Diode symbol

Each symbol file:
- Uses the IR subcircuit name (preserving fidelity to the IR)
- Contains a template that references the IR name via `@model`
- Is dialect-agnostic (works with any spice simulator)

### Bridge Files

Bridge files map IR subcircuit names to PDK-specific subcircuits. In this example, they map IR names to themselves (pass-through), but in real usage they would map to actual PDK names like `sky130_fd_pr__nfet_01v8`.

**bridge_xyce.spice** - For Xyce simulator:
```spice
.subckt nmos d g s b l=0.15 w=1.0 m=1.0
    Xprim d g s b nmos l={l} w={w} m={m}
.ends nmos
```

**bridge_ngspice.spice** - For Ngspice simulator:
```spice
.subckt nmos d g s b l=0.15 w=1.0 m=1.0
    Xprim d g s b nmos l={l} w={w} m={m}
.ends nmos
```

## Using in Xschem

### Option 1: Using Docker

1. **Validate symbols (optional):**
   ```bash
   python3 test_xschem_symbols.py
   ```

2. **Open xschem with symbols:**
   ```bash
   ./open_in_xschem.sh
   ```
   This will launch xschem in docker with the symbols accessible.

### Option 2: Local Installation

1. **Place symbol files in your xschem library:**
   ```bash
   cp output/*.sym ~/.xschem/xschem_library/
   ```

2. **In your testbench, include the appropriate bridge file:**
   - For Xyce: `.include "bridge_xyce.spice"`
   - For Ngspice: `.include "bridge_ngspice.spice"`

3. **Use the symbols in your schematic:**
   - The symbols reference IR names (e.g., `nmos_1v0`, `pmos_1v0`)
   - The bridge file maps these to the actual subcircuit definitions
   - Switch simulators by changing the bridge file include

## Key Features

- **Dialect-Agnostic Symbols**: Symbols work with any spice simulator
- **IR Name Preservation**: Symbols use IR subcircuit names, maintaining fidelity
- **Multiple Dialect Support**: Generate bridge files for xyce, ngspice, spice, etc.
- **Easy Simulator Switching**: Change simulators by swapping bridge files

## Extending to Real PDKs

In a real scenario with PDK files:

1. The IR would contain PDK-specific subcircuit names (e.g., `sky130_fd_pr__nfet_01v8`)
2. The bridge file would map IR names to these PDK names:
   ```spice
   .subckt nmos d g s b l=0.15 w=1.0 m=1.0
       Xprim d g s b sky130_fd_pr__nfet_01v8 l={l} w={w} m={m}
   .ends nmos
   ```
3. The bridge file would also include PDK model files:
   ```spice
   .include "sky130_fd_pr/models/sky130_xyce.lib"
   ```

