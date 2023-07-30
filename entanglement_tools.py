import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from typing import *
from TransformerWF import *
import cirq
from numpy import linalg as LA
import sympy


def index_to_state(idx, qubit_num):
	"""
	"""
	tot_qubits = 2 ** qubit_num

	state = [int(el) for el in bin(idx)[2:]]

	state = [0] * (qubit_num - len(state)) + state

	return np.array(state)


def state_to_index(state):
	"""
	"""
	string_state = "".join([str(el) for el in state])

	return int(string_state, 2)


def give_reduced_density_eigs(
		state: np.array,
		region: Tuple[int, int],
) -> np.array:
	"""
	given a state in the standard basis, retrun the eigenvalues of the
	reduced density matrix.

	We compute this using a schmidt decomposition

	region is inclusive
	"""
	"""
	First, decompose our state into the form of 

	\psi = \sum b_{i, j} e_i x f_j

	where e_i and f_j span the two regions

	This means we rewrite our state
	"""

	total_qubit_num = int(round(np.log2(len(state))))

	internal_length = region[-1] - region[0]
	external_length = total_qubit_num - internal_length
	new_state = np.zeros((2 ** internal_length, 2 ** external_length), dtype=np.complex128)

	for idx1 in range(2 ** internal_length):
		basis_state1 = index_to_state(idx1, internal_length)

		for idx2 in range(2 ** external_length):
			basis_state2 = index_to_state(idx2, external_length)

			full_basis_state = list(basis_state2[:region[0]]) + list(basis_state1) + list(basis_state2[region[0]:])

			# now map to the standard basis notation
			state_index = state_to_index(full_basis_state)

			# add this value to the reformulation of the state
			new_state[idx1, idx2] += state[state_index]

	"""
	Now we simply perform SVD
	"""
	sqrt_eigs = LA.svd(
		new_state,
		compute_uv=False,
	)

	"""
	finally, return the eigs of the density matrix
	"""
	return np.abs(sqrt_eigs) ** 2


def give_nth_renyi(
		n: int,
		state: np.array,
		region: Tuple[int, int],
):
	"""
	Use the schmidt decompositon to compute the nth renyi (equivalent)
	Tr[rho_a^n]
	"""
	rho_eigs = give_reduced_density_eigs(
		state=state,
		region=region,
	)

	return np.sum(rho_eigs ** n)


def give_entanglement_entropy(
		state: np.array,
		region: Tuple[int, int],
):
	"""
	Use the schmidt decompositon to compute the nth renyi (equivalent)
	Tr[rho_a^n]
	"""
	rho_eigs = give_reduced_density_eigs(
		state=state,
		region=region,
	)

	s1 = -np.sum(rho_eigs * np.log(rho_eigs))

	return s1


def get_explicit_transformer_state(
		decoder,
		qubit_num: int,
		assert_real=True,
		kink_picture=False,
) -> np.array:
	"""

	assert_real allows us to remove the phase

	"""
	full_state_in_standard_basis = []

	for i in range(2 ** qubit_num):
		state = [int(x) for x in bin(i)[2:]]
		state = [0] * (qubit_num - len(state)) + state

		if kink_picture:
			state = [int(state[i] != state[i+1]) for i in range(len(state)-1)]

		log_probs, total_phases = decoder.evaluate_state(
			np.array([state])
		)

		if assert_real:
			full_state_in_standard_basis.append(
				np.exp(float(log_probs[0]) / 2)
			)
		else:
			full_state_in_standard_basis.append(
				np.exp(float(log_probs[0]) / 2) * np.exp(1j * float(total_phases[0]))
			)
	if kink_picture:
		return np.array(full_state_in_standard_basis)/np.sqrt(2)
	return np.array(full_state_in_standard_basis)


