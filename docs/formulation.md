### Papers referenced:
- OpenAI CLIP - Learning Transferable Visual Models From Natural Language Supervision
- OpenAI SAE - Scaling and evaluating sparse autoencoders
- Hugo Fry - Towards Multimodal Interpretability: Learning Sparse Interpretable Features in Vision Transformers  

### Problem Statement
CLIP is a joint embedding model that is trained to maximize cosine similarity of (text, image) ebeddings. The result is an embedding model that encodes into a high dimensional vector space representing concepts and ideas that arise jointly in natural language and vision.  
SAE's are encoder-decoder networks that encode into some latent space, enforcing sparsity condition onto that latent rep. via a condition baked into the architecture or added into the loss term. I ultimately trained a model that implemented both types a la OpenAI SOA methods.  
I will seek to train an SAE on a smaller subset of the imagenet-1k dataset, imagenet-mini (50,000 images), and deliver a model in which latent space neurons correspond to interpretable features in the images. 

Hugo Fry's Towards Multimodal Interp. served as the high level outline for this project, and as a starting place for model architecture and hyperparameter settings. However, I ultimately deviated quite a bit from his approach for a couple reasons:
- His codebase on github was not parsable in 2 days 
- He implements L1 loss - current SOA is Topk (in OpenAI paper) 
- Limited compute meant needed faster convergence (Topk)

### Architecture

Ultimately, I settled on Eluether AI's open source SAE library, which implements TopK SAE's for LLM's in a relatively quick-to-learn fashion. Because their SAE's are configured for text embedding models, there was some overhead required to make it functional on the CLIPVisionModel, which is discussed later. While I played with training the SAE at various points throughout the CLIPVisionModel, the output of each CLIP Resblock were vectors of length 768, and only taking the CLS token in the residual sequence (something that took me a long time to realize was not implemented in Eleuther AI), meant each SAE recieved a 1x768 vec and took the following form: 
<p align="center">
Linear(768, expansion_factor*768) -> ReLU() -> TopK(k) -> Linear(expansion_factor*768, 768)
</p>


with the Decoder initialized to the transpose of the encoder for faster convergence.

The loss recommended by Gao et al. in Scaling and evaluating sparse autoencoders is 


$$  \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;   \mathcal{L} = MSE + \alpha*aux_k $$


Which is the MSE of the reconstruction error of the decoder from the latent space, and $ aux_k $ is the auxilliary reconstruction error by using $ aux_k $  dead latents (a dead latent is defined as a neuron that has't fired in #_threshold tokens, or in this case, batches, because each batch has only one CLS token). My undertanding is that this encourages a larger latent feature space by punishing convergence to a small subset of features. Practically in the code, the MSE is measured as the Fraction of Variance Unexplained, and $ \alpha $ is set to a value ~ 0.03. 































  
