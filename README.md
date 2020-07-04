# Cycle-GAN
In this repo, a notebook aims at implementing the Cycle GAN model based on the original paper https://arxiv.org/abs/1703.10593 and present an overview of the theory presented in the paper. 

## Introduction and background

First let us recall the idea of regular GAN model and then go on the how the CycleGAN differs. 
Given some training data $x$ and noise $z$. More details are given in our PDF report but the regular GAN optimization program is the following : 

The generator minimizes $log(1 – D(G(z)))$ and the discriminator maximizes $log D(x) + log(1 – D(G(z)))$. It has been shown that using least squares losses for this problem yields better results. Therefore we can use the following program. The generator will minimize $(D(G(z)) – 1)^2$ and the discriminator will minimize $(D(x) – 1)^2 + (D(G(z)))^2$.


The Cycle GAN architecture contains two GAN each one being made of a generator and a discriminator. 


The goal is to learn mapping functions between two
domains $\mathcal{X}$ and $\mathcal{Y}$ given training samples $\{x_i\}_{i=1, \dots, N}$ where $x_i$ ∈ $\mathcal{X}$ and $\{y_j\}_{j=1, \dots, N}$ $y_j$ ∈ $\mathcal{Y}$. Let us denote $x ∼ p_{data}(x)$ and $y ∼ p_{data}(y)$. 

The model will include two mappings $G : \mathcal{X} → \mathcal{Y}$ and $F : \mathcal{Y} → \mathcal{X}$.

In addition, the authors introduce two adversarial discriminators $D_X$ and $D_Y$ , where $D_X$ aims to distinguish between images ${x}$ and translated
images ${F(y)}$; in the same way, $D_Y$ aims to discriminate between ${y}$ and ${G(x)}$. 

Now the cycle gan architecture becomes different because of the loss it will consider. The objective contains two types of terms: adversarial losses (same as regular gan) for matching the distribution of generated images to the data distribution in the target domain and **cycle consistency losses **to prevent the learned mappings $G$ and $F$ from contradicting each other.


For the mapping function $G : X → Y$ and its discriminator $D_Y$ , we express the objective as:

$$
L_{GAN}(G, D_Y , X, Y ) = \mathbb{E}_{y∼p_{data}(y)} [log D_Y (y)]
+ \mathbb{E}_{x∼p_{data}(x)} [log(1 − D_Y (G(x))]
$$


Similarly for $F : Y → X$, the objective is : 

$$
L_{GAN}(F, D_X , Y, X ) = \mathbb{E}_{x∼p_{data}(x)} [log D_X (x)]
+ \mathbb{E}_{y∼p_{data}(y)} [log(1 − D_X (G(y))]
$$


And the novel part of the loss is :

$$
L_{cyc}(G, F) = \mathbb{E}_{x∼p_{data}(x)} [||F(G(x)) − x||_1]
+ \mathbb{E}_{y∼p_{data}(y)} [||G(F(y)) − y||_1]
$$


Finally, the total loss the Cycle GAN will optimize is the following one : 

$$
L(G, F, D_X, D_Y ) =L_{GAN}(G, D_Y , X, Y )
+ L_{GAN}(F, D_X, Y, X)
+ λL_{cyc}(G, F)
$$


Where $\lambda$ is parameter to control how much ty cyclic loss term will count.




However, as for the standard GAN, we will use least square losses for the GAN losses part. This is detailed later on.



## Optimization process 

After defining the model, we use our data to optimize the weights of the neural network during the training process. To do that, each component of the cycle gan will have a loss function to minimize.  By using the same previous notations we define the loss function for each generator and discriminator:
$$L_{G} = \mathbb{E}_{x∼p_{data}(x)}[(DY(G(x))-1)^2] + \lambda L_{cycle}$$ \\
$$L_{F} = \mathbb{E}_{y∼p_{data}(y)}[(DX(F(y))-1)^2] + \lambda L_{cycle}$$ \\

Where $L_{cycle} = \mathbb{E}_{x∼p_{data}(x)}[\parallel F(G(x)) - x \parallel_1] + \mathbb{E}_{y∼p_{data}(y)}[\parallel G(F(y)) - y \parallel_1]$ \\
\\
$$L_{DX} = \mathbb{E}_{x∼p_{data}(x)}[(DX(x) - 1)^2] + \mathbb{E}_{y∼p_{data}(y)}[DX(F(y))^2]$$ \\
$$L_{DY} = \mathbb{E}_{y∼p_{data}(y)}[(DY(y) - 1)^2] + \mathbb{E}_{x∼p_{data}(x)}[DY(G(x))^2]$$


## Results 

Results are present in the notebook. For computational reasons, a simpler neural network architecture is used in this implementation than in the original paper. See original paper for fancy and impressive applications of this technique. 





