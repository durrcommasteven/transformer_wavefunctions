# transformer_wavefunctions

A transformer model designed to learn the ground state of a Hamiltonian

For a colloquial introduction to the use of language models for modeling quantum systems (and transformers specifically), see the writeup on 
[my blog](https://durrcommasteven.github.io/blog/transformer-quantum-states/)

 - `TFIM_Visualizations.ipynb` A jupyter notebook with visuals comparing transformer results to results found using DMRG
 - `TransformerWF.py` Defining the transformer model, with functions to estimate Renyi entropies and Von Neumann entropy
 - `final_DMRG_comparison.ipynb` A jupyter notebook to compute values through DMRG
 - `final_transformer_analysis.ipynb` Compute expected quantities using transformer wave functions
 - `final_version_transformer_MHS.ipynb` A notebook training Transformer wavefunctions on the modified Haldane-Shastry model,
    as in https://arxiv.org/pdf/1701.04844.pdf (MHS)
 - `final_version_transformer_TFIM.ipynb` A notebook training Transformer wavefunctions on the transverse field ising model (TFIM)
 
 
