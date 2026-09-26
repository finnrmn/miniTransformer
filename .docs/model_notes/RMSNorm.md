# Normalization: RMSNorm in pre-norm placement
```
@inproceedings{zhang2019rmsnorm,
  title         = {Root Mean Square Layer Normalization},
  author        = {Zhang, Biao and Sennrich, Rico},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {32},
  year          = {2019},
  url           = {https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html},
  eprint        = {1910.07467},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
Two changes at the normalization site are bundled. Layer normalization shifts the input vector $x \in \mathbb{R}^d$ by subtracting its mean $\mu$ and scales it using $\sigma$ before applying the learned gain $\gamma$ and bias $\beta$. _~zhang2019rmsnorm_ hypothesize that the scaling invariance is the primary driver of training stability and discard the mean-centering operation as well as the bias $\beta$. Instead, Root Mean Square Normalization (RMSNorm) scales the vector solely by its root mean square $\mathrm{RMS}(x)$:

$$
	\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}, \qquad
	\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$
where $\gamma \in \mathbb{R}^d$ denotes the learned gain parameter. The saving is one statistic per vector. The paper reports a training speedup of 7--9% for a Transformer at equal translation quality (26.8 against 26.6 BLEU on newstest2014) and measures nothing at inference _~zhang2019rmsnorm_. The second change is placement. The original block normalizes the sum of the residual stream and the sublayer output (post-norm), pre-norm normalizes the input of each sublayer instead and adds one final normalization after the last block. GPT-2 had adopted the placement empirically. At initialization, the gradient norm at the last block scales as $O(d\sqrt{\ln d})$ under post-norm but as $O(d\sqrt{\ln d / L})$ under pre-norm, which the authors give as the reason post-norm training requires a small initial learning rate and a warm-up stage that pre-norm can omit. Both changes leave the residual stream unbroken from the embedding to the output head.
