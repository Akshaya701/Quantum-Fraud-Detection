import os

from qiskit.circuit.library import (
    z_feature_map,
    zz_feature_map,
    pauli_feature_map
)


RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


feature_maps = {

    "z_feature_map":

        z_feature_map(
            feature_dimension=2,
            reps=2
        ),

    "zz_feature_map":

        zz_feature_map(
            feature_dimension=2,
            reps=2,
            entanglement="linear"
        ),

    "pauli_feature_map":

        pauli_feature_map(
            feature_dimension=2,
            reps=2,
            paulis=[
                "Z",
                "ZZ",
                "X"
            ]
        )
}


for name, circuit in feature_maps.items():

    print(
        f"\n{name}"
    )

    print(
        circuit.draw(
            output="text"
        )
    )

    try:

        circuit.draw(
            output="mpl",
            filename=os.path.join(
                RESULTS_DIR,
                f"{name}_circuit.png"
            )
        )

    except Exception as error:

        print(
            "Could not save circuit image:",
            error
        )


print(
    "\nFeature-map circuits generated."
)