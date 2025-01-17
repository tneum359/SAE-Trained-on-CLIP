Please reference analysis.ipynb and train.ipyng for code. This section explains specific modifications to open source code and things to note for reproducability. 
I decided to go with Eleuther AI's codebase for SAE implementation because Hugo Frys code wasn't parsable in 2 days and Gytis Daujotas' code was implemented for a different purpose than simply training an SAE for interpretaibility. Additionally, topK SAE's are the current SOA, not L1 loss. Eleuther AI's codebase is relatively clean but has the downside of being implemented for LLM's, which required the following modifications. 
```python
#trainer.py
#...
def hook(module: nn.Module, inputs, outputs):
            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            #only grab the CLS token while keeping batch and dim
            outputs = outputs[:, 0, :]
            name = module_to_name[module]

            #store in dict for evaluating over SAE
            output_dict[name] = outputs
```
Define custome hook function that only grabs the CLS token of the sequence of embeddings. Previously it was evaluating the SAE over the entire sequence of text embeddings and evaluating the SAE right away over the output. Instead, store them in a dict for the entire batch, then run SAE and backprop over all outputs in ouput_dict
```python
#trainer.py
#...
class CLIPImageDataset(Dataset):
    """
    A map-style dataset that returns a single (image, label) pair
    in the format your model expects.
    """
    def __init__(self, hf_dataset, clip_processor):
        super().__init__()
        self.hf_dataset = hf_dataset   
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        # Process the image with the CLIP processor
        proc = self.clip_processor(images=sample["image"], return_tensors="pt")
        # shape [1, 3, 224, 224], so we squeeze out the batch dimension
        pixel_values = proc["pixel_values"].squeeze(0)  # -> [3,224,224]
        label = sample.get("label", -1)
        return {
            "input_ids": pixel_values,
            "labels": label
        }
```
Adjust dataset for CLIPVisionModel input. Eleuther AI's code is specific for "input_id's" key in dataset, so use this hacky way of keeping that key while inserting the pixel values. Then adjust token counts within the trainer.py code for dead_neuron tracking. 

With these modifications (or just using the available github), the training routines should be reproducible. 

### Hyperparameters
With more compute I would probably perform a sweep over hyperparameters, but instead I just trained three models with the following configurations:


Model 1 (trained on entire sequence of tokens){
CLIP location: layer 9
lr: 10e-4
k: 32
expansion faction: 8 (num latents 6144)
batch: 10
normalize_decoder=True
auxk_alpha: 0
}
Loss = 0.88

Model 2 (trained only on CLS token){
CLIP location: layer 9
lr: 10e-3
k: 32
expansion faction: 8 (num latents 6144)
batch: 100
normalize_decoder=False
auxk_alpha: 1/32
}
Loss = 0.54

Model 3 (trained only on CLS token){
CLIP location: layer 11
lr: 5e-4
k: 64
expansion faction: 4 (num latents 3072)
batch: 500
normalize_decoder=False
auxk_alpha: 1/32
}
Loss = 0.29


Evidently, training only on the CLS token seemed to make the biggest loss performance (and actually impose interpretability in the feature space), while decreasing the latent space and increasing competition among neurons (higher k) hypothetically resulted in more superposition to reconstruct the input. With more time, I would probably further reduce the latent embedding size and increase k - I hypothesize that two low of a k value can make the model get stuck in a local optimum. Increasing the range of neurons working the together to reconstruct the input can encourage richer feature representations. Additionally, actually non-zeroing auxk alpha likely made a huge difference in promoting all neurons to acquire some representational power (no neurons were actually "dead" (never firing) when evaluated over the entire dataset). The difference in the first and subsequent models can likely be attributed to some combination of training on the CLS token, reducing latent space, and adding in auxilliary loss. With more compute, I would perofrm a more fine-grained hyperparameter sweep. 




















