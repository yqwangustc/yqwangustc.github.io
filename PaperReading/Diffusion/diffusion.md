# From VAE to Diffusion Model

## Varational Autoencoder (VAE)
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


## Denoise Diffusion Probablistic Model (DDPM)

In the DDPM, we start from $\boldsymbol x_0$, whose distribution is unknown. At each step $t$, a diffusion process is used: 

$$
    \boldsymbol x_{t} = \alpha_t\,\boldsymbol x_{t-1} + \beta_t\,\boldsymbol \varepsilon_t
$$

where $\boldsymbol\varepsilon_t$ draws from zero-mean unit-variance Gaussion distribution. Additionally, $\alpha_t^2 + \beta_t^2 = 1$. We have the following attributes regarding $\boldsymbol x_t$:

- **Forward Process**: given $\boldsymbol x_{t-1}$, it is straightfoward to know that $p(\boldsymbol x_t | \boldsymbol x_{t-1})$ follow a Gaussin distribution with $\alpha_t \, \boldsymbol x_{t-1}$ as mean and $\beta_t^2 \boldsymbol I$ as the variance.

- **Fast Forward Process**: A nice property of DDPM is that the conditional distribution of $\boldsymbol x_{t}$ given $\boldsymbol x_{0}$ can be calculated explicitly without going through the recrusive process, i.e., 

  $$ 
  \begin{align}
    \boldsymbol{x}_t &= \alpha_t \boldsymbol{x}_{t-1} + \beta \boldsymbol{\varepsilon_t} \nonumber\\ 
    & = \alpha_t \alpha_{t-1}...\alpha_1 \boldsymbol{x}_0 + (\alpha_t ... \alpha_2)\beta_1 \boldsymbol \varepsilon_1 + ... + \beta_t \boldsymbol\varepsilon_t
  \end{align}  
  $$

  Except for the first term in Eq. (6), each term is a zero-mean, unit-variance Gaussion noise, therefore, Eq (6) can be also written as: 

  $$
  \begin{align}
    \boldsymbol{x}_t = \overline{\alpha}_t \boldsymbol x_0 + \overline{\beta}_t \overline{\boldsymbol{\varepsilon}_t}
  \end{align}
  $$
  where $\overline{\alpha}_t = \prod_{\tau=1}^{t}\alpha_\tau$, $\overline{\beta}_t = \sqrt{1- \overline\alpha_t^2}$ and $\overline{\boldsymbol\varepsilon}_t$ is again a zero-mean, uni-variance Gaussin.
 
- **Reverse Process**: However, we don't know the conditional distribution of $\boldsymbol{x}_{t-1}$ given $\boldsymbol x_{t}$. We only know that for small enough $\beta_t$, it is a still a Gaussian distribution. We use neural network (with parameter $\theta$) to estimate the mean and variance, given $\boldsymbol x_t$ and $t$, i.e., 
  
  $$ p(\boldsymbol x_{t-1} | \boldsymbol x_{t}) = \mathcal{N}(\boldsymbol x_{t-1}; \boldsymbol \mu_{\theta}(\boldsymbol x_t, t), \boldsymbol \Sigma_{\theta}(\boldsymbol x_t, t))$$

- **Conditional Reverse Process**: Though we don't know the explicit form of reverse probability, a nice property of diffusion model is that $p(\boldsymbol x_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)$ is a Gaussian distribution. This can be proved by the following deduction: 

    $$
    \begin{align*}
        p(\boldsymbol x_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0) &= 
        \frac{p(\boldsymbol{x}_t, \boldsymbol{x}_{t-1} | \boldsymbol{x}_0)}{p(\boldsymbol x_t | \boldsymbol{x}_0)} = \frac{p(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}) p (\boldsymbol x_{t-1} | \boldsymbol{x}_0)}{p(\boldsymbol x_t | \boldsymbol{x}_0)}
    \end{align*}
    $$

    Since we know that $p(\boldsymbol x_\tau | \boldsymbol{x}_0)$ is a Gaussian distribution $\mathcal N(\boldsymbol{x}_\tau; \overline\alpha_{\tau} \boldsymbol{x}_0, \overline{\beta}_\tau \boldsymbol{I} )$ for $\tau > 1$ from **Forward Process**, we then have the following equations:

    $$
    \begin{align*}
        p(\boldsymbol x_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0) \sim 

        \exp(-\frac{1}{2\beta_t^2}(\boldsymbol{x}_t - \alpha_t \boldsymbol{x}_{t-1})^2)
        \cdot
        \exp(-\frac{1}{2\overline\beta_{t-1}^2}(\boldsymbol{x}_{t-1} - \overline\alpha_{t-1}\boldsymbol{x}_0)^2)\cdot
        \exp(\frac{1}{2\overline\beta_t^2}(\boldsymbol{x}_t - \overline\alpha_t \boldsymbol{x}_0)^2)
    \end{align*}
    $$

    Note that the above equation can be written as:

    $$
        p(\boldsymbol x_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0) \sim
        \exp(-\frac{1}{2}(a \boldsymbol{x}_{t-1}^2 - 2b \boldsymbol x_{t-1}+c))
    $$
    with (note that $\alpha_t^2 + \beta_t^2 = 1$, $\overline\alpha_t^2 + \overline\beta_t^2=1$ and $\overline\alpha_t = \overline\alpha_{t-1} \alpha_t$)

    $$
    \begin{align*}
        a &= \frac{\alpha_t^2}{\beta_t^2} + \frac{1}{\overline\beta_{t-1}^2} 
          = \frac{\alpha_t^2(1-\overline\alpha_{t-1}^2) + \beta_t^2}{\beta_t^2 (1- \overline\alpha_{t-1}^2)} = \frac{1-\overline\alpha_t^2}{\beta_t^2 (1- \overline\alpha_{t-1}^2)}
        \\
        b &= \frac{\alpha_t}{\beta_t^2} \boldsymbol{x}_t + \frac{\overline\alpha_{t-1}}{\overline\beta_{t-1}^2} \boldsymbol{x}_0 \\

        \frac{b}{a} &= \frac{\alpha_t - \overline\alpha_t^2}{1- \overline\alpha_t^2} \boldsymbol x_t + \frac{\beta_t^2 \overline\alpha_{t-1}}{1- \overline\alpha_t^2} \boldsymbol{x}_0
    \end{align*}
    $$

    Therefore, we can see that $p(\boldsymbol x_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)$ is again Gaussian, and 

    $$
    \boldsymbol x_{t-1} = \frac{\alpha_t - \overline\alpha_t^2}{1- \overline\alpha_t^2} \boldsymbol x_t + \frac{\beta_t^2 \overline\alpha_{t-1}}{1- \overline\alpha_t^2} \boldsymbol{x}_0 + \frac{\beta_t^2 (1 - \overline\alpha_{t-1}^2)}{1- \overline\alpha_t^2} \boldsymbol\varepsilon
    $$

    Using the **Fast forward** property, we know that 

    $$
    \boldsymbol{x}_0 = \frac{1}{\overline\alpha_t} (\boldsymbol{x}_t  - \overline\beta_t \overline{\boldsymbol{\varepsilon}_t})
    $$