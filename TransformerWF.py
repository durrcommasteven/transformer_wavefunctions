import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from typing import *
import sympy


def convert_affine_chebyshev(k):
	"""
	Given T_k(2x-1), return a list of the coefficients
	from order 0 to order k
	"""

	x = sympy.Symbol("x")

	poly = np.polynomial.chebyshev.chebval(2 * x - 1, [0] * k + [1])

	coefficients = []

	for n in range(k + 1):
		coeff = sympy.diff(poly, x, n).subs({x: 0}) / np.math.factorial(n)

		coefficients.append(coeff)

	return coefficients


def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates


def generate_positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
							np.arange(d_model)[np.newaxis, :],
							d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	return tf.cast(angle_rads, dtype=tf.float32)


def create_look_ahead_mask(size):
	mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask


def pad_states(states):
	"""
    states should be a tensor or array of shape
    (batch_size, sequence_length)
    we pad on either side of the sequence with zeros to produce
    an array of shape (batch_size, sequence_length+2)
    """
	return np.pad(states, (1, 0), mode='constant')[1:]


class TransformerDecoder(tf.keras.Model):
	def __init__(
			self,
			num_heads: int,
			key_dim: int,
			value_dim: int,
			embedding_dim: int,
			dictionary_size: int,
			decoding_reps: int,
			width: int,
			sequence_length: int,
			depth: int = 2,
			final_temperature: float = 1,
			random_positional_encoding: bool = False,
			trainable_positional_encoding: bool = False,
			conv_feed_forward: Optional[Tuple[int, int]] = (3, 1),
			dropout: float = 0,
			attention_dropout: float = 0,
			name: str = None,
	):
		"""
		Define a module a decoding transformer
		The 'plain' part here is that this does not use any encoder.
		It does not have a second layer of attention taking in keys and
		values from an encoder's embedding.

		IMPORTANT: sequence length includes the start token, and indicates the
		length of the sequences used during training. This means that if we set
		sequence length to be 11, we'll be considering states with maximum length 10.
		Technically we dont need a fixed maximum sequence length, but this makes sense
		to have in the context of hamiltoninans.

		As in the attention is all you need paper, we apply dropout as follows:
		"Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the
		sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
		positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
		P_drop = 0.1."
		We also apply dropout within the self attention layers.

		Args:
			num_heads: int,
			key_dim: int,
			value_dim: int,

			embedding_dim: int,
			dictionary_size: int,

			decoding_reps: int, The number of repetitions of attention,
				followed by a feed-forward network.
			width: int, The width of the feed forward network if it is
				chosen to be fully-connected
			sequence_length: int, the sequence length used during training
				This includes the start-token
			depth: int = 2, The depth of the feed-forward networks used

			random_positional_encoding: bool, Whether the positional encoding
				is taken to be random, or the version used in
				'attention is all you need'
			trainable_positional_encoding: bool, Whether the positional
				embedding is trainable

			conv_feed_forward: Optional[Tuple[int, int]] = (3, 1),
				Either None, indicating we use a dense network, or
				a tuple of This gives us the parameters for our conv1d network:
				conv_feed_forward = (filters, kernel_size)

			name: str = None, The name of this model
		"""
		super().__init__(name=name)
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.value_dim = value_dim
		self.embedding_dim = embedding_dim
		self.dictionary_size = dictionary_size
		self.decoding_reps = decoding_reps
		self.width = width
		self.depth = depth
		self.sequence_length = sequence_length
		self.random_positional_encoding = random_positional_encoding
		self.conv_feed_forward = conv_feed_forward
		self.final_temperature = final_temperature
		self.droput_rates = (dropout, attention_dropout)

		"""
		Define the functional model below
		"""

		"""
		Define the positional encodings
		"""
		if random_positional_encoding:
			self.positional_encodings = tf.Variable(
				tf.random.normal(shape=(sequence_length, embedding_dim)),
				trainable=trainable_positional_encoding
			)
		else:
			self.positional_encodings = tf.Variable(
				generate_positional_encoding(sequence_length, embedding_dim),
				trainable=trainable_positional_encoding
			)

		"""
		Define the embeddings
		"""
		self.embeddings = tf.keras.layers.Embedding(
			input_dim=dictionary_size,
			output_dim=embedding_dim,
			trainable=True,
		)

		"""
		Define the attention layers
		Note we do not apply dropout here.
		"""
		self.attention_layers = [
			tfa.layers.MultiHeadAttention(
				head_size=key_dim,
				num_heads=num_heads,
				output_size=value_dim,
				dropout=attention_dropout,
				use_projection_bias=True,
				return_attn_coef=False,
				name=f"attention_{idx}",
			) for idx in range(decoding_reps)
		]

		"""
		These each take in query, keys, and values
		define the layers to define each of these
		"""

		"""
		Define the feed-forward components
		"""
		if conv_feed_forward is None:
			self.feed_forward = [
				[
					tf.keras.layers.Dense(
						width, activation="relu", use_bias=True,
						kernel_initializer='glorot_uniform',
						bias_initializer='zeros', kernel_regularizer=None,
						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
						bias_constraint=None, name=f"fully-connected_{ff_idx}_layer_{layer_idx}"
					) for layer_idx in range(depth - 1)
				] + [
					tf.keras.layers.Dense(
						embedding_dim, activation=None, use_bias=True,
						kernel_initializer='glorot_uniform',
						bias_initializer='zeros', kernel_regularizer=None,
						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
						bias_constraint=None, name=f"fully-connected_{ff_idx}_layer_{depth - 1}"
					)
				]
				for ff_idx in range(decoding_reps)
			]
		else:
			filters, kernel_size = conv_feed_forward

			self.feed_forward = [
				[
					tf.keras.layers.Conv1D(
						filters=filters, kernel_size=kernel_size, activation="relu", use_bias=True,
						kernel_initializer='glorot_uniform',
						bias_initializer='zeros', kernel_regularizer=None,
						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
						bias_constraint=None, name=f"fully-connected_{ff_idx}_layer_{layer_idx}"
					) for layer_idx in range(depth - 1)
				] + [
					tf.keras.layers.Dense(
						embedding_dim, activation=None, use_bias=True,
						kernel_initializer='glorot_uniform',
						bias_initializer='zeros', kernel_regularizer=None,
						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
						bias_constraint=None, name=f"fully-connected_{ff_idx}_layer_{depth - 1}"
					)
				]
				for ff_idx in range(decoding_reps)
			]

		"""
		Define the normalizing layer
		"""
		self.layer_norms = [
			[
				tf.keras.layers.LayerNormalization(
					axis=-1,
					trainable=False
				) for idx1 in range(2)
			] for idx2 in range(decoding_reps)
		]

		"""
		define a dropout layer
		This is applied to the encodings after embedding them 
		"""
		self.dropout_layer = tf.keras.layers.Dropout(dropout)

		"""
		Define the final linear layer which will give us logits
		
		Here we define a glorot normal initializer with std scaled by the 
		temperature
		"""
		self.final_kernel_initializer = tf.keras.initializers.RandomNormal(
			mean=0.0,
			stddev=np.sqrt(
				2 / (self.embedding_dim + self.dictionary_size)
			) / np.sqrt(self.final_temperature)
		)

		self.final_layer = tf.keras.layers.Dense(
			dictionary_size, activation=tf.nn.log_softmax, use_bias=False,
			kernel_initializer=self.final_kernel_initializer, #'glorot_uniform',
			name="final_linear_layer",
		)

		self.phase_layer = tf.keras.layers.Dense(
			1, activation=None, use_bias=True,
			kernel_initializer='glorot_uniform',
			name="final_linear_layer",
		)

		self.build((None, self.sequence_length))

	def call(self, inputs):
		"""
		inputs should be a tensor of tokens

		the only constraint I place on it is that

		inputs.shape should be of the form

		(..., batchsize, length)

		This applies positional embeddings
		"""
		"""
		Combine these into a functional model
		"""

		embeddings = self.embeddings(inputs)

		assert embeddings.shape[-2] <= self.sequence_length, (
			f"expected sequence length <= {self.sequence_length}, recieved length {embeddings.shape[-2]}"
		)

		residual_stream = self.dropout_layer(
			embeddings + self.positional_encodings[:embeddings.shape[-2], :]
		)

		mask = create_look_ahead_mask(embeddings.shape[-2])

		for idx in range(self.decoding_reps):
			attention_output = self.attention_layers[idx](
				[residual_stream, residual_stream], mask=mask
			)

			attention_output = self.dropout_layer(attention_output)

			residual_stream = self.layer_norms[idx][0](
				residual_stream + attention_output
			)

			"""
			Creating the feed-forward layers
			"""
			ff_argument = residual_stream

			for feed_forward_layer in self.feed_forward[idx]:
				ff_argument = feed_forward_layer(ff_argument)

			ff_argument = self.dropout_layer(ff_argument)

			residual_stream = self.layer_norms[idx][1](
				ff_argument + residual_stream
			)

		"""
		mapping to logits
		"""
		logits = self.final_layer(residual_stream)

		phase = np.pi * tf.nn.softsign(
			tf.squeeze(self.phase_layer(residual_stream), axis=-1)
		)

		return logits, phase

	def autoregressive_sampling(self, initial_states):
		"""
		A function to implement autoregressive sampling
		run this within gradient tape

		initial_states should be of shape
		(batch_size, intial_length)

		typically, initial_length will be 1

		we'll cut off the initial and final spins before returning
		"""
		batch_size = initial_states.shape[0]

		autoregressive_spin_states = np.copy(initial_states)

		autoregressive_phases = tf.zeros(batch_size)
		log_probabilities = tf.zeros(batch_size)

		for idx in range(self.sequence_length):
			conditional_logits, phases = self.call(
				inputs=autoregressive_spin_states
			)

			"""
			if we are past the start token, accumulate the phase
			"""
			if autoregressive_spin_states.shape[-1] > 1:
				autoregressive_phases += phases[:, -1]

			"""
			if we aren't at the final sequence element, select dictionary items
			"""
			if idx < self.sequence_length - 1:
				choices = tf.random.categorical(
					conditional_logits[:, -1, :],
					num_samples=1,
					dtype=tf.int32,
				)

				autoregressive_spin_states = np.hstack(
					[autoregressive_spin_states, choices]
				)

				log_probabilities += tf.gather(
					conditional_logits[:, -1, :],
					tf.reshape(choices, -1),
					batch_dims=1,
				)
		return (
			np.array(autoregressive_spin_states)[..., 1:],
			log_probabilities,
			autoregressive_phases
		)

	def evaluate_state(self, state, z_symmetric=False):
		"""
		state should be of shape (batch_size, sequence_length)

		return log_probability, phase

		"""
		state = state.astype(np.int32)

		batch_size = state.shape[0]
		length = state.shape[1]

		conditional_logits, phases = self.call(
			inputs=pad_states(state),
		)

		"""
		Collect the total phases (aside from the final phase)
		and the conditional
		"""
		total_phases = tf.reduce_sum(phases[:, 1:], -1)

		log_probs = tf.reduce_sum(
			tf.gather(
				conditional_logits[:, :-1, :],
				state,
				batch_dims=2
			),
			axis=-1,
		)

		if z_symmetric:
			flog_probs, ftotal_phases = self.evaluate_state(1-state, z_symmetric=False)

			total_prob = (tf.math.exp(flog_probs)+tf.math.exp(log_probs))/2

			log_probs = tf.math.log(total_prob)

		return log_probs, total_phases

	def evaluate_gradients(
			self,
			batch_size,
			local_energy_function,
			return_energy=True,
			reps=1,
			z_symmetric=False,
	):
		"""
		local energy function is a function of states, and the decoder

		return the gradients

		I take the initial states to be zeros

		this uses (C6) in https://arxiv.org/pdf/2002.02973.pdf
		"""

		initial_states = np.zeros((batch_size, 1))

		if False:
			with tf.GradientTape() as tape:
				loss = 0
				for rep in range(reps):
					"""
					generate samples
					"""
					samples, log_probs, phases = self.autoregressive_sampling(
						initial_states
					)

					"""
					I dont want gradients passing through E loc 
					So I'll use their numpy counterparts
					"""
					E_loc_real_part, E_loc_imag_part = local_energy_function(
						samples,
						self
					)

					if tf.is_tensor(E_loc_real_part):
						E_loc_real_part = E_loc_real_part.numpy()
					if tf.is_tensor(E_loc_imag_part):
						E_loc_imag_part = E_loc_imag_part.numpy()

					E_estimate_real_part = np.mean(E_loc_real_part)
					E_estimate_imag_part = np.mean(E_loc_imag_part)

					"""
		
					Construct the loss function
		
					For each state this becomes
		
					(1/2)*D_log_p*Re[E_{loc} - E]
					+D_phase*Im[E_{loc} - E]
		
					"""

					loss += (2 / batch_size) * tf.reduce_sum(
						(1 / 2) * log_probs * (E_loc_real_part - E_estimate_real_part)
					)

					loss += (2 / batch_size) * tf.reduce_sum(
						phases * (E_loc_imag_part - E_estimate_imag_part)
					)

				loss /= reps

				print(loss)

		"""
		generate samples
		"""
		all_samples = []
		E_estimate_real_parts = []
		E_estimate_imag_parts = []

		for rep in range(reps):
			samples, log_probs, phases = self.autoregressive_sampling(
				initial_states
			)
			all_samples.append(samples)

			"""
			I dont want gradients passing through E loc 
			So I'll use their numpy counterparts
			"""
			E_loc_real_part, E_loc_imag_part = local_energy_function(
				samples,
				self
			)

			if tf.is_tensor(E_loc_real_part):
				E_loc_real_part = E_loc_real_part.numpy()
			if tf.is_tensor(E_loc_imag_part):
				E_loc_imag_part = E_loc_imag_part.numpy()

			E_estimate_real_part = np.mean(E_loc_real_part)
			E_estimate_imag_part = np.mean(E_loc_imag_part)

			E_estimate_real_parts.append(E_estimate_real_part)
			E_estimate_imag_parts.append(E_estimate_imag_part)

		with tf.GradientTape() as tape:
			"""
			evaluate samples
			"""
			loss = 0

			for sample_idx, samples in enumerate(all_samples):
				log_probs, phases = self.evaluate_state(
					samples,
				)

				if z_symmetric:
					total_flip_log_probs, total_flip_phases = self.evaluate_state(
						1-samples,
					)

				"""
	
				Construct the loss function
	
				For each state this becomes
	
				(1/2)*D_log_p*Re[E_{loc} - E]
				+D_phase*Im[E_{loc} - E]
	
				"""
				E_estimate_real_part = E_estimate_real_parts[sample_idx]
				E_estimate_imag_part = E_estimate_imag_parts[sample_idx]

				loss += (2 / batch_size) * tf.reduce_sum(
					(1 / 2) * log_probs * (E_loc_real_part - E_estimate_real_part)
				)

				loss += (2 / batch_size) * tf.reduce_sum(
					phases * (E_loc_imag_part - E_estimate_imag_part)
				)

				if z_symmetric:
					z_symmetric_reg = 10

					loss += z_symmetric_reg*tf.reduce_sum((total_flip_log_probs-log_probs)**2)

			loss /= reps

		derivatives = tape.gradient(loss, self.trainable_weights)

		if return_energy:
			return derivatives, (np.mean(E_estimate_real_parts), np.mean(E_estimate_imag_parts))

		return derivatives

	def compute_nth_renyi(
			self,
			n: int,
			batch_size: int,
			region: Tuple[int, int],
			reps: int = 1
	) -> float:
		"""
		Compute a direct-sampling estimate of the nth renyi entropy.

		Note that
		\rho_a = \sum_b <\sigma_b | \rho |\sigma_b>

		therefore for n=3
		\Tr[\rho_a^3] = \sum_{a1, a2, a3} \sum_{b1, b2, b3}
			<\sigma_a1 \sigma_b1 | \rho |\sigma_a2 \sigma_b1> *
			<\sigma_a2 \sigma_b2 | \rho |\sigma_a3 \sigma_b2> *
			<\sigma_a3 \sigma_b3 | \rho |\sigma_a1 \sigma_b3>

		For n>2, it proceeds like this

		Therefore, we use important sample to obtain an estimate of this quantity.
		This involves sampling n*batchsize samples according to p(\sigma_a, \sigma_b),
		producing {(\sigma^i_a, \sigma^i_b)}, and {(\sigma^{i+1}_a, \sigma^i_b)}
		Evaluate the wave functions of these states, and compute

		<
		(\prod_i \psi(\sigma^i_a, \sigma^i_b) \psi*(\sigma^{i+1}_a, \sigma^i_b)) /
		(\prod_i \psi(\sigma^i_a, \sigma^i_b) \psi*(\sigma^i_a, \sigma^i_b))
		>

		Should return Tr[\rho_a^n]

		Args:
			n:
			batch_size:
			region:
			reps:

		Returns:

		"""
		assert type(n) == int, f"expected n to be an int, recieved {type(n)}"
		assert n >= 2, f"expected n>=2, recieved n={n}"
		assert region[0] < region[1]

		total_batch_size = batch_size * n
		seed_state = np.zeros((total_batch_size, 1))

		expectations = []

		for _ in range(reps):
			# sample autoregressively
			states, log_probabilities, autoregressive_phases = self.autoregressive_sampling(
				seed_state
			)

			# reshape these to have the correct shape

			states = np.reshape(states, newshape=(batch_size, n, self.sequence_length-1))
			# states will have shape (batch_size, n, sequence_length-1)

			log_probabilities = np.reshape(log_probabilities, newshape=(batch_size, n))
			autoregressive_phases = np.reshape(autoregressive_phases, newshape=(batch_size, n))
			# log_probabilities and autoregressive_phases will have shape (batch_size, n)

			permuted_states = np.roll(
				states,
				axis=1,
				shift=-1,
			)

			"""
			Now reassign those elements which are outside of the region
			"""

			permuted_states[:, :, :region[0]] = states[:, :, :region[0]]
			permuted_states[:, :, region[1]:] = states[:, :, region[1]:]

			permuted_log_probabilities, permuted_total_phases = self.evaluate_state(
				np.reshape(
					permuted_states,
					newshape=(batch_size * n, self.sequence_length-1)
				)
			)

			permuted_log_probabilities = np.reshape(
				permuted_log_probabilities,
				newshape=(batch_size, n)
			)
			permuted_total_phases = np.reshape(
				permuted_total_phases,
				newshape=(batch_size, n)
			)

			"""
			Now compute the current contribution to the renyi estimate
			"""
			all_ratios = np.exp(
				np.sum(permuted_log_probabilities - log_probabilities, -1)/2
			)
			all_phases = np.exp(
				1j * np.sum(-permuted_total_phases + autoregressive_phases, -1)
			)

			batch_expectation = np.mean(all_ratios * all_phases)

			expectations.append(batch_expectation)

		return np.mean(expectations)

	def compute_von_neumann_estimate(
			self,
			region: Tuple[int, int],
			order: int = 7,
			total_sample_size: int = 10_000,
			max_total_batchsize: int = 1_000,
	) -> float:
		"""

		here we compute the nth renyis

		Note that we in computing the nth renyi's, the wave functions are evaluated over
		n * batchsize samples. We therefore need to limit this number in order to
		ensure the number of samples does not exceed the available memory.
		We therefore have max_total_batchsize.

		Args:
			region: 
			batch_size: 
			reps: 
			order: 
			total_sample_size: 
			max_total_batchsize: 

		Returns:

		"""

		# first define the expansion coefficients
		coefficients = [
			np.log(2) - 1 / 2, np.log(2) - 3 / 4
		]
		coefficients += [(-1) ** (k + 1) / (k * (k ** 2 - 1)) for k in range(2, order + 1)]

		"""
		now we compute the nth renyi's. To do this, we want to adjust the batchsize here
		so that the maximum total batchsize is limited to a manageable size. 
		"""

		# create Tr[rho^0] (not used) and Tr[rho^1] (=1)
		nth_renyis = [1, 1]

		for n in range(2, order + 1):
			# compute our batchsize and reps such that we sample at least

			batch_size = int((max_total_batchsize / n) + 1)
			reps = int((total_sample_size / batch_size) + 1)

			nth_renyi = self.compute_nth_renyi(
				n=n,
				batch_size=batch_size,
				region=region,
				reps=reps
			)

			nth_renyis.append(nth_renyi)

		# now return the estimate for the von neumann entropy
		entropy_estimate = 0

		for n in range(order + 1):
			contribution = 0

			chebyshev_terms = convert_affine_chebyshev(n)

			for nth_cheb_coeff, nth_order in zip(chebyshev_terms, np.real(nth_renyis)):
				contribution += nth_cheb_coeff * nth_order

			entropy_estimate += coefficients[n] * contribution

		return float(entropy_estimate)
