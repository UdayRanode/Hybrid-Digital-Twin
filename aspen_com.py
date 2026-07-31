import os
from pathlib import Path
import win32com.client as win32
import surrogate_model

ASPEN_VERSION = "Apwn.Document.40.0"
ASPEN_VISIBLE = True
SUPPRESS_DIALOGS = True

PROJECT_DIR = Path(__file__).resolve().parent
ASPEN_FILE = PROJECT_DIR / "SoC.bkp"

FEED_STREAM = "MIXED"
REACTOR_BLOCK = "PFR"
OUTLET_STREAM = "S6"

# =============================================================================
# SECTION 2 : ASPEN CONNECTION
# =============================================================================

def connect_to_aspen():
    print("\nLaunching Aspen Plus...")
    try:
        aspen = win32.Dispatch(ASPEN_VERSION)
        print("Aspen COM object created successfully.")
        return aspen
    except Exception as error:
        raise RuntimeError(f"Unable to launch Aspen Plus.\n{error}")

def open_simulation(aspen):
    print("\nOpening Aspen simulation...")
    print("Aspen file path:")
    print(ASPEN_FILE)
    print("Exists:", ASPEN_FILE.exists())
    if not ASPEN_FILE.exists():
        raise FileNotFoundError(f"Aspen file not found:\n{ASPEN_FILE}")
    try:
        print("Opening:", ASPEN_FILE)
        print("Exists:", ASPEN_FILE.exists())
        aspen.InitFromArchive2(str(ASPEN_FILE))
        aspen.Visible = ASPEN_VISIBLE
        if SUPPRESS_DIALOGS:
            aspen.SuppressDialogs = True
        print("Simulation loaded successfully.")
    except Exception as error:
        raise RuntimeError(f"Unable to open Aspen simulation.\n{error}")


def close_simulation(aspen):
    print("\nClosing Aspen Plus...")
    try:
        aspen.Close()
    except Exception:
        pass
    print("Aspen session closed.")

# =============================================================================
# SECTION 3 : ASPEN TREE PATHS
# =============================================================================

MIXED_TEMP_NODE = r"\Data\Streams\MIXED\Input\TEMP\MIXED"
MIXED_PRESSURE_NODE = r"\Data\Streams\MIXED\Input\PRES\MIXED"

MIXED_CO2_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\CO2"
MIXED_H2_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\HYDROGEN"
MIXED_CO_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\CO"
MIXED_CH4_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\METHANE"
MIXED_C2H4_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\ETHYLENE"
MIXED_C3H6_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\PROPYLEN"
MIXED_H2O_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\WATER"
MIXED_C4H8_NODE = r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\1-BUT-01"

S6_CO2_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\CO2"
S6_H2_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\HYDROGEN"
S6_CO_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\CO"
S6_CH4_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\METHANE"
S6_C2H4_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\ETHYLENE"
S6_C3H6_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\PROPYLEN"
S6_H2O_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\WATER"
S6_C4H8_NODE = r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\1-BUT-01"

# =============================================================================
# SECTION 4 : GENERIC ASPEN UTILITY FUNCTIONS
# =============================================================================

def read_node(aspen, node_path):
    try:
        node = aspen.Tree.FindNode(node_path)
        if node is None:
            raise ValueError(f"Node not found:\n{node_path}")
        return node.Value
    except Exception as error:
        raise RuntimeError(
            f"Unable to read Aspen node:\n{node_path}\n\n{error}"
        )


def write_node(aspen, node_path, value):
    try:
        node = aspen.Tree.FindNode(node_path)
        if node is None:
            raise ValueError(f"Node not found:\n{node_path}")
        node.Value = value
    except Exception as error:
        raise RuntimeError(
            f"Unable to write Aspen node:\n{node_path}\n\n{error}"
        )


def run_simulation(aspen):
    print("\nRunning Aspen simulation...")
    try:
        aspen.Engine.Run2()
        print("Simulation completed successfully.")
    except Exception as error:
        raise RuntimeError(f"Simulation failed.\n{error}")

# =============================================================================
# SECTION 5 : READ REACTOR FEED STREAM (MIXED)
# =============================================================================

def read_mixed_stream(aspen):
    print("\nReading reactor feed stream (MIXED)...")
    mixed_stream = {
        "temperature": read_node(aspen, MIXED_TEMP_NODE),
        "pressure": read_node(aspen, MIXED_PRESSURE_NODE),
        "CO2": read_node(aspen, MIXED_CO2_NODE),
        "H2": read_node(aspen, MIXED_H2_NODE),
        "CO": read_node(aspen, MIXED_CO_NODE),
        "CH4": read_node(aspen, MIXED_CH4_NODE),
        "C2H4": read_node(aspen, MIXED_C2H4_NODE),
        "C3H6": read_node(aspen, MIXED_C3H6_NODE),
        "H2O": read_node(aspen, MIXED_H2O_NODE),
        "C4H8": read_node(aspen, MIXED_C4H8_NODE),
    }
    print("Reactor feed successfully read.")
    return mixed_stream

# =============================================================================
# SECTION 6 : LOAD SURROGATE MODEL
# =============================================================================

def load_surrogate_model():
    print("\nLoading surrogate model...")
    try:
        model, scaler_X, scaler_y = surrogate_model.load_surrogate()
        print("Surrogate model loaded successfully.")
        return model, scaler_X, scaler_y
    except Exception as error:
        raise RuntimeError(f"Unable to load surrogate model.\n\n{error}")

# =============================================================================
# SECTION 7 : RUN SURROGATE REACTOR
# =============================================================================

def run_surrogate_reactor(model, scaler_X, scaler_y, mixed_stream):
    print("\nRunning surrogate reactor...")
    try:
        outlet_stream = surrogate_model.predict_reactor(
            model=model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            T_C=mixed_stream["temperature"],
            P_bar=mixed_stream["pressure"],
            F_CO2_in=mixed_stream["CO2"],
            F_H2_in=mixed_stream["H2"],
        )
        print("Surrogate prediction completed successfully.")
        return outlet_stream
    except Exception as error:
        raise RuntimeError(f"Surrogate reactor prediction failed.\n\n{error}")

# =============================================================================
# SECTION 8 : WRITE REACTOR OUTLET STREAM (S6)
# =============================================================================

def write_reactor_outlet(aspen, outlet_stream):
    print("\nWriting reactor outlet stream (S6)...")
    try:
        write_node(aspen, S6_CO2_NODE, outlet_stream["CO2"])
        write_node(aspen, S6_H2_NODE, outlet_stream["H2"])
        write_node(aspen, S6_CO_NODE, outlet_stream["CO"])
        write_node(aspen, S6_CH4_NODE, outlet_stream["CH4"])
        write_node(aspen, S6_C2H4_NODE, outlet_stream["C2H4"])
        write_node(aspen, S6_C3H6_NODE, outlet_stream["C3H6"])
        write_node(aspen, S6_H2O_NODE, outlet_stream["H2O"])
        write_node(aspen, S6_C4H8_NODE, outlet_stream["C4H8"])
        print("Reactor outlet stream written successfully.")
    except Exception as error:
        raise RuntimeError(
            f"Unable to write reactor outlet stream.\n\n{error}"
        )

# =============================================================================
# SECTION 9 : EXECUTE ASPEN FLOWSHEET
# =============================================================================

def execute_flowsheet(aspen):
    print("\nExecuting Aspen flowsheet...")
    try:
        run_simulation(aspen)
        print("Flowsheet executed successfully.")
    except Exception as error:
        raise RuntimeError(f"Failed to execute Aspen flowsheet.\n\n{error}")

# =============================================================================
# SECTION 10 : MAIN PROGRAM
# =============================================================================

def main():
    aspen = None
    try:
        aspen = connect_to_aspen()
        open_simulation(aspen)
        run_simulation(aspen)
        model, scaler_X, scaler_y = load_surrogate_model()
        mixed_stream = read_mixed_stream(aspen)
        outlet_stream = run_surrogate_reactor(
            model, scaler_X, scaler_y, mixed_stream
        )
        write_reactor_outlet(aspen, outlet_stream)
        execute_flowsheet(aspen)
        print("\nDigital Twin simulation completed successfully.")
    except Exception as error:
        print("\nERROR:")
        print(error)
    finally:
        if aspen is not None:
            close_simulation(aspen)

# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()