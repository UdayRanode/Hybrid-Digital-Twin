# =============================================================================
# ASPEN PLUS COM INTERFACE
# Hybrid Digital Twin for CO2 Hydrogenation to Light Olefins
# =============================================================================


import os
from pathlib import Path

import win32com.client as win32

import surrogate_model
surrogate_model.load_surrogate()


ASPEN_VERSION = "Apwn.Document.40"     # Aspen Plus V14

ASPEN_VISIBLE = True

SUPPRESS_DIALOGS = True


PROJECT_DIR = Path(__file__).resolve().parent

ASPEN_FILE = PROJECT_DIR / "SoC bkp.bkp"



FEED_STREAM = "MIXED"

REACTOR_BLOCK = "PFR"

OUTLET_STREAM = "S6"

# =============================================================================
# SECTION 2 : ASPEN CONNECTION
# =============================================================================

def connect_to_aspen():
    """
    Launch Aspen Plus and return the COM Application object.
    """

    print("\nLaunching Aspen Plus...")

    try:

        aspen = win32.Dispatch(ASPEN_VERSION)

        aspen.Visible = ASPEN_VISIBLE

        if SUPPRESS_DIALOGS:
            aspen.SuppressDialogs = True

        print("Aspen Plus launched successfully.")

        return aspen

    except Exception as error:

        raise RuntimeError(
            f"Unable to launch Aspen Plus.\n{error}"
        )


def open_simulation(aspen):
    """
    Open the Aspen Plus backup (.bkp) file.
    """

    print("\nOpening Aspen simulation...")

    if not ASPEN_FILE.exists():

        raise FileNotFoundError(
            f"Aspen file not found:\n{ASPEN_FILE}"
        )

    try:

        aspen.InitFromArchive2(str(ASPEN_FILE))

        print("Simulation loaded successfully.")

    except Exception as error:

        raise RuntimeError(
            f"Unable to open Aspen simulation.\n{error}"
        )


def close_simulation(aspen):
    """
    Close Aspen Plus.
    """

    print("\nClosing Aspen Plus...")

    try:

        aspen.Close()

    except Exception:

        pass

    print("Aspen session closed.")

    # =============================================================================
# SECTION 3 : ASPEN TREE PATHS
# =============================================================================

# -----------------------------------------------------------------------------
# MIXED Stream (Reactor Feed)
# -----------------------------------------------------------------------------

MIXED_TEMP_NODE = (
    r"\Data\Streams\MIXED\Input\TEMP\MIXED"
)

MIXED_PRESSURE_NODE = (
    r"\Data\Streams\MIXED\Input\PRES\MIXED"
)

# -----------------------------------------------------------------------------
# MIXED Stream Component Molar Flows (kmol/hr)
# -----------------------------------------------------------------------------

MIXED_CO2_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\CO2"
)

MIXED_H2_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\HYDROGEN"
)

MIXED_CO_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\CO"
)

MIXED_CH4_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\METHANE"
)

MIXED_C2H4_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\ETHYLENE"
)

MIXED_C3H6_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\PROPYLEN"
)

MIXED_H2O_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\WATER"
)

MIXED_C4H8_NODE = (
    r"\Data\Streams\MIXED\Output\MOLEFLOW\MIXED\1-BUT-01"
)

# -----------------------------------------------------------------------------
# Reactor Outlet Stream (S6)
# -----------------------------------------------------------------------------

S6_CO2_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\CO2"
)

S6_H2_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\HYDROGEN"
)

S6_CO_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\CO"
)

S6_CH4_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\METHANE"
)

S6_C2H4_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\ETHYLENE"
)

S6_C3H6_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\PROPYLEN"
)

S6_H2O_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\WATER"
)

S6_C4H8_NODE = (
    r"\Data\Streams\S6\Output\MOLEFLOW\MIXED\1-BUT-01"
)

# =============================================================================
# SECTION 4 : GENERIC ASPEN UTILITY FUNCTIONS
# =============================================================================

def read_node(aspen, node_path):
    """
    Read a value from an Aspen Plus tree node.

    Parameters
    ----------
    aspen : COM object
        Aspen Plus COM Application object.

    node_path : str
        Aspen tree path.

    Returns
    -------
    float
        Value stored in the specified Aspen node.
    """

    try:

        node = aspen.Tree.FindNode(node_path)

        if node is None:

            raise ValueError(
                f"Node not found:\n{node_path}"
            )

        return node.Value

    except Exception as error:

        raise RuntimeError(
            f"Unable to read Aspen node:\n"
            f"{node_path}\n\n"
            f"{error}"
        )


def write_node(aspen, node_path, value):
    """
    Write a value to an Aspen Plus tree node.

    Parameters
    ----------
    aspen : COM object
        Aspen Plus COM Application object.

    node_path : str
        Aspen tree path.

    value : float
        Value to be written.
    """

    try:

        node = aspen.Tree.FindNode(node_path)

        if node is None:

            raise ValueError(
                f"Node not found:\n{node_path}"
            )

        node.Value = value

    except Exception as error:

        raise RuntimeError(
            f"Unable to write Aspen node:\n"
            f"{node_path}\n\n"
            f"{error}"
        )


def run_simulation(aspen):
    """
    Execute the Aspen Plus simulation.
    """

    print("\nRunning Aspen simulation...")

    try:

        aspen.Engine.Run2()

        print("Simulation completed successfully.")

    except Exception as error:

        raise RuntimeError(
            f"Simulation failed.\n{error}"
        )

# =============================================================================
# SECTION 5 : READ REACTOR FEED STREAM (MIXED)
# =============================================================================

def read_mixed_stream(aspen):
    """
    Read reactor feed conditions from the MIXED stream.

    Returns
    -------
    dict
        Dictionary containing reactor inlet conditions.

        {
            "temperature": float,
            "pressure": float,
            "CO2": float,
            "H2": float,
            "CO": float,
            "CH4": float,
            "C2H4": float,
            "C3H6": float,
            "H2O": float,
            "C4H8": float,
        }
    """

    print("\nReading reactor feed stream (MIXED)...")

    mixed_stream = {

        "temperature": read_node(
            aspen,
            MIXED_TEMP_NODE,
        ),

        "pressure": read_node(
            aspen,
            MIXED_PRESSURE_NODE,
        ),

        "CO2": read_node(
            aspen,
            MIXED_CO2_NODE,
        ),

        "H2": read_node(
            aspen,
            MIXED_H2_NODE,
        ),

        "CO": read_node(
            aspen,
            MIXED_CO_NODE,
        ),

        "CH4": read_node(
            aspen,
            MIXED_CH4_NODE,
        ),

        "C2H4": read_node(
            aspen,
            MIXED_C2H4_NODE,
        ),

        "C3H6": read_node(
            aspen,
            MIXED_C3H6_NODE,
        ),

        "H2O": read_node(
            aspen,
            MIXED_H2O_NODE,
        ),

        "C4H8": read_node(
            aspen,
            MIXED_C4H8_NODE,
        ),

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

        raise RuntimeError(
            f"Unable to load surrogate model.\n\n{error}"
        )

# =============================================================================
# SECTION 7 : RUN SURROGATE REACTOR
# =============================================================================

def run_surrogate_reactor(
        model,
        scaler_X,
        scaler_y,
        mixed_stream,
):
    

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

        raise RuntimeError(
            f"Surrogate reactor prediction failed.\n\n{error}"
        )

# =============================================================================
# SECTION 8 : WRITE REACTOR OUTLET STREAM (S6)
# =============================================================================

def write_reactor_outlet(aspen, outlet_stream):
    """
    Write the surrogate reactor outlet stream to Aspen.

    Parameters
    ----------
    aspen : COM object
        Aspen Plus COM application object.

    outlet_stream : dict
        Reactor outlet stream returned by predict_reactor().

    Returns
    -------
    None
    """

    print("\nWriting reactor outlet stream (S6)...")

    try:

        write_node(
            aspen,
            S6_CO2_NODE,
            outlet_stream["CO2"],
        )

        write_node(
            aspen,
            S6_H2_NODE,
            outlet_stream["H2"],
        )

        write_node(
            aspen,
            S6_CO_NODE,
            outlet_stream["CO"],
        )

        write_node(
            aspen,
            S6_CH4_NODE,
            outlet_stream["CH4"],
        )

        write_node(
            aspen,
            S6_C2H4_NODE,
            outlet_stream["C2H4"],
        )

        write_node(
            aspen,
            S6_C3H6_NODE,
            outlet_stream["C3H6"],
        )

        write_node(
            aspen,
            S6_H2O_NODE,
            outlet_stream["H2O"],
        )

        write_node(
            aspen,
            S6_C4H8_NODE,
            outlet_stream["C4H8"],
        )

        print("Reactor outlet stream written successfully.")

    except Exception as error:

        raise RuntimeError(
            f"Unable to write reactor outlet stream.\n\n{error}"
        )

# =============================================================================
# SECTION 9 : EXECUTE ASPEN FLOWSHEET
# =============================================================================

def execute_flowsheet(aspen):
    """
    Execute the Aspen Plus flowsheet after updating the reactor outlet.

    Parameters
    ----------
    aspen : COM object
        Aspen Plus COM application object.

    Returns
    -------
    None
    """

    print("\nExecuting Aspen flowsheet...")

    try:

        run_simulation(aspen)

        print("Flowsheet executed successfully.")

    except Exception as error:

        raise RuntimeError(
            f"Failed to execute Aspen flowsheet.\n\n{error}"
        )

# =============================================================================
# SECTION 10 : MAIN PROGRAM
# =============================================================================

def main():
    """
    Main function to execute the Aspen Plus - Surrogate Model integration.
    """

    aspen = None

    try:

        # ---------------------------------------------------------------------
        # Connect to Aspen and open simulation
        # ---------------------------------------------------------------------
        aspen = connect_to_aspen()

        open_simulation(aspen)

        # ---------------------------------------------------------------------
        # Load trained surrogate model
        # ---------------------------------------------------------------------
        model, scaler_X, scaler_y = load_surrogate_model()

        # ---------------------------------------------------------------------
        # Read reactor inlet stream
        # ---------------------------------------------------------------------
        mixed_stream = read_mixed_stream(aspen)

        # ---------------------------------------------------------------------
        # Run surrogate reactor
        # ---------------------------------------------------------------------
        outlet_stream = run_surrogate_reactor(
            model,
            scaler_X,
            scaler_y,
            mixed_stream,
        )

        # ---------------------------------------------------------------------
        # Write predicted reactor outlet to Aspen
        # ---------------------------------------------------------------------
        write_reactor_outlet(
            aspen,
            outlet_stream,
        )

        # ---------------------------------------------------------------------
        # Execute remaining Aspen flowsheet
        # ---------------------------------------------------------------------
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
