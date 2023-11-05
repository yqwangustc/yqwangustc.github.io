# From VAE to Diffusion Model

In the normal auto-encoder (AE) model, for a data distribution $p(\boldsymbol{x})$, we first encode $\boldsymbol x$ using $q(\boldsymbol z|\boldsymbol x)$ and decode it using $p(\boldsymbol x|\boldsymbol z)$. We need to optimize the loglikelihood of $\boldsymbol x$ for a given encoding function $q(\boldsymbol z | \boldsymbol x)$. This gives the following loss function:

$$\mathcal{L}_{\mathtt ae}(x) = \mathbf{E}_{q(\boldsymbol{z} | \boldsymbol{x})} [ p(\boldsymbol{x} | \boldsymbol{z})] + P_{\alpha} $$

where $P_{\alpha}$ is regularization term. However it is unclear how to devise such reguarlaization term in principle. 

Based on AE, varational AE (VAE) derivied the loss function in a probablistic manner. We starts from $\log p(\boldsymbol x)$: 

$$
\begin{align}
\log p(\boldsymbol x) &= \log \int p(\boldsymbol x, | \boldsymbol z) p(\boldsymbol{z}) d\boldsymbol{z} \\
    & = \log \int \frac{p(\boldsymbol{x}|\boldsymbol{z}) p(\boldsymbol{z})}{q(\boldsymbol{z} | \boldsymbol{x})} q(\boldsymbol{z} | \boldsymbol{x}) d\boldsymbol{z}  \\
    & \geq \int \log \frac{p(\boldsymbol{x} | \boldsymbol{z})p(\boldsymbol{z})}{q(\boldsymbol{z}|\boldsymbol{x})} q(\boldsymbol{z}|\boldsymbol{x}) d\boldsymbol{z} \\
    & = \mathbf{E}_q [\log p(\boldsymbol{x} | \boldsymbol{z})] - \int \log\frac{p(\boldsymbol{z})}{q(\boldsymbol{z}|\boldsymbol{x})} q(\boldsymbol{z} | \boldsymbol{x}) d\boldsymbol{z} \\
    & = \mathbf{E}_q [\log p(\boldsymbol{x} | \boldsymbol{z})]  + \mathrm{KL}(q(\boldsymbol{z} | \boldsymbol{x}) || p(\boldsymbol{z}))
\end{align}
$$

The first term in the above equation is the log-likelihood of decoder output, while the second term minimize the KL divergence between encoder output and the target encoder distribution. Now the $P_\alpha$ in the Eq (1) has a probabilistic definition. 

