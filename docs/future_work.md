Targets if I were to continue on this project:
- Train on a larger dataset with more compute (full imagenet-1k with higher latent space and full hyperparameter sweep)
- Investigate more videos with queryscoring a la CLIP Hitchhiker
    - I would approach this by first training a model on full imagenet-1k to hopefully map a wider range of concepts. I fear that imagenet-mini might have been a weird subset of images for generalizing to arbitrary videos
    - Test on more videos with more streamlined comparisons. Think of better way of evaluating then comparing to imagenet images 
    - Experiment with better text descriptions for calculating scores for weighted average. For the single example I examined, the text "construction hardhat dirt truck forest operator", even if the SAE model had the power to extract meaningful features, might not have been specific enough
- Expand the evaluation to calculate image entropy and add that information into the histogram of sparsity vs avg. activation, investigate how that effects high vs low band frequency clusters 
      
  