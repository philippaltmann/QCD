import pennylane as qml


class PX(qml.operation.Operation):
    """ Wrapper for Phased-X Operation, as used/defined in
    https://github.com/google-research/google-research/tree/master/rl4circopt """

    # Define how many wires the operator acts on in total.
    num_wires = 1

    # Define what differentiation method to use (here: parameter-shift)
    grad_method = "A"

    def __init__(self, phi1, phi2, wires):

        # checking the inputs --------------
        shape1 = qml.math.shape(phi1)
        shape2 = qml.math.shape(phi2)
        if len(shape1) > 1:
            raise ValueError(f"Phi1: Expected a scalar angle; got angle of shape {shape1}.")
        if len(shape2) > 1:
            raise ValueError(f"Phi2: Expected a scalar angle; got angle of shape {shape2}.")

        # extract all wires that the operator acts on
        all_wires = qml.wires.Wires(wires)

        # The parent class expects all trainable parameters to be fed as positional
        # arguments, and all wires acted on fed as a keyword argument.
        super().__init__(phi1, phi2, wires=all_wires)

    @property
    def num_params(self):
        # if it is known before creation, define the number of parameters to expect here,
        # which makes sure an error is raised if the wrong number was passed
        return 2

    @staticmethod
    def compute_decomposition(phi1, phi2, wires):

        op_z_p = qml.exp(qml.PauliZ(wires), 1j * phi2)
        op_x = qml.exp(qml.PauliX(wires), 1j * phi1)
        op_z_m = qml.exp(qml.PauliZ(wires), -1j * phi2)

        return [op_z_p, op_x, op_z_m]

    def adjoint(self):
        return PX(-self.parameters[0], self.parameters[1], self.wires)
