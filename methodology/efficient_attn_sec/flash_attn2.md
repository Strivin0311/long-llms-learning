## Forward

*you can find the tutorial of how to implement flash-attn forward kernel with triton [here](../../notebooks/tutorial_triton.ipynb)*

### For standard attn forward:

$$
\begin{cases}
\begin{align} P &= \mathrm{mask}(QK^{\mathrm{T}} + bias)  \in \mathbb{R}^{N\times N} \notag\end{align}\\
\begin{align} A &= \mathrm{softmax}_{row\text{-}wise}(P) = \mathrm{diag}(l)^{-1}S  \in \mathbb{R}^{N\times N}\notag\end{align}, \quad where \; l = \mathrm{rowsum}(S) \in \mathbb{R}^{N}, \quad S = \exp{(P \!- \mathrm{rowmax}(P))} \in \mathbb{R}^{N\times N} \\ 
\begin{align} O &= AV \in \mathbb{R}^{N\times d} \notag\end{align}
\end{cases}
$$

given $Q,K,V \in \mathbb{R}^{N\times d}$

### For flash-attn forward:

#### step0. the basic attention row-decomposition:

$$\begin{cases}
\begin{align} P &= \left[ P_1\quad P_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\quad where\; P_i = \mathrm{mask}(QK_i^{\mathrm{T}} + bias) \in \mathbb{R}^{B_q\times B_k}, Q \in \mathbb{R}^{B_q\times d}, K_i \in \mathbb{R}^{B_k\times d},\;i \in \{1,2\} \notag\end{align}\\
\begin{align} m &= \max\big( \mathrm{rowmax}(P_1), \mathrm{rowmax}(P_2) \big) \in \mathbb{R}^{B_q} \notag\end{align}\\
\begin{align} S &= \left[ S_1\quad S_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\quad where\; S_i = \exp(P_i \!- m) \in \mathbb{R}^{B_q\times B_k}, \;i \in \{1,2\}  \notag\end{align}\\
\begin{align} l &= \mathrm{rowsum}(S_1) + \mathrm{rowsum}(S_2) \in \mathbb{R}^{B_q} \notag\end{align}\\
\begin{align} A &= \left[ A_1\quad A_2 \right] = \mathrm{diag}(l)^{-1}\left[ S_1\quad S_2 \right] \in \mathbb{R}^{B_q\times 2B_k} \notag\end{align}\\
\begin{align} O &= \left[ A_1\quad A_2 \right] \left[ \begin{matrix}V_1\\ V_2 \end{matrix}\right] = \mathrm{diag}(l)^{-1} \big( S_1V_1 + S_2V_2 \big) \in \mathbb{R}^{B_q\times d} \notag\end{align}\\
\end{cases}
$$
  
#### step1. the online-softmax attention:
$$\text{base}: \begin{cases}
\begin{align} m_1 &= \mathrm{rowmax}(P_1) \in \mathbb{R}^{B_q},\quad S_1 = \exp(P_1\!- m_1) \in \mathbb{R}^{B_q\times B_k}\notag\end{align}\\
\begin{align} l_1 &= \mathrm{rowsum}(S_1)\in \mathbb{R}^{B_q},\quad A_1 = \mathrm{diag}(l_1)^{-1}S_1\in \mathbb{R}^{B_q\times B_k}  \notag\end{align}\\
\begin{align} O_1 &= A_1V_1\in \mathbb{R}^{B_q\times d} \notag\end{align}\\
\end{cases}
$$

$$\text{update}: \begin{cases}
\begin{align} m_2 &= \max(m_1, \mathrm{rowmax}(P_2)) \in \mathbb{R}^{B_q},\quad S_2 = \exp(P_2\!- m_2) \in \mathbb{R}^{B_q\times B_k}\notag\end{align}\\
\begin{align} l_2 &= \delta_m l_1 + \mathrm{rowsum}(S_2)\in \mathbb{R}^{B_q},\quad A_2 = \mathrm{diag}(l_2)^{-1}S_2\in \mathbb{R}^{B_q\times B_k}  \notag\end{align}\\
\begin{align} O_2 &= \mathrm{diag}(l_1/l_2)^{-1}\delta_m O_1 + A_2V_2 \in \mathbb{R}^{B_q\times d} \notag\end{align}\\
\end{cases}
$$

where $\delta_m := \exp(m_1\!-m_2)$

#### step2: flash-attn forward algorithm with tiling (double-loop):
* the outer loop runs through $i := 1 \rightarrow N_q$ for each block of $Q_i$ to compute $O_i$,  where $N_q = \lceil\frac{N}{B_q}\rceil$

    $$\text{in one i-th outer iteration}: \begin{cases}
    \begin{align} \text{load}\; Q_i \in \mathbb{R}^{B_q\times d}\; \text{from HBM to SRAM}\notag\end{align}\\
    \begin{align} \text{initialize}\; \tilde{O_i}^{(0)} = (0)_{B_q\times d} \in \mathbb{R}^{B_q\times d},\; l_i^{(0)} = (0)_{B_q} \in \mathbb{R}^{B_q},\; m_i^{(0)} = (-\infty)_{B_q} \in \mathbb{R}^{B_q}  \notag\end{align}\\
    \begin{align} \text{loop over}\; j := 1 \rightarrow N_k\; \text{for each j-th inner iteration} \notag\end{align}\\
    \begin{align} \text{compute}\; O_i = \mathrm{diag}(l_{i}^{(N_k)})^{-1} \tilde{O_i}^{(N_k)}\in \mathbb{R}^{B_q\times d}\; \text{and write it to HBM to return as output} \notag\end{align}\\
    \begin{align} \text{compute}\; \text{LSE}_i = m_i^{(N_k)} + \log(l_i^{(N_k)})\in \mathbb{R}^{B_q}\; \text{and write it to HBM to save for backward} \notag\end{align}\\
    \end{cases}
    $$

    where $\text{LSE}(\bold{x}) := \log\big(\sum\limits_{i=1}^n \exp(x_i)\big) = \max(\bold x) + \text{LSE}(\bold{x}-\max(\bold x)),\; \bold x \in \mathbb{R}^{n}$, and $\tilde{O_i}$ is the un-normalized $O_i$, i.e. $O_i = \mathrm{diag}(l_{i})^{-1}\tilde{O_i}$

* in which each inner loop goes across $j := 1 \rightarrow N_k$ for each block of $K_j,V_j$ to update $\tilde{O_i}^{(j)}, l_i^{(j)}, m_i^{(j)}$, where $N_k = \lceil\frac{N}{B_k}\rceil$

    $$\text{in one j-th inner iteration}: \begin{cases}
    \begin{align} \text{load}\; K_j, V_j \in \mathbb{R}^{B_k\times d}\; \text{from HBM to SRAM} \notag\end{align}\\
    \begin{align} \text{compute}\; P_{i}^{(j)} = \text{mask}(Q_iK_j^{\mathrm T} + bias) \in \mathbb{R}^{B_q\times B_k} \notag\end{align}\\
    \begin{align} \text{update}\; m_i^{(j)} &= \max\big(m_i^{(j-1)}, \mathrm{rowmax}(P_{i}^{(j)})\big) \in \mathbb{R}^{B_q} \notag\end{align}\\
    \begin{align} \text{compute}\;S_i^{(j)} &= \exp(P_i^{(j)}\!- m_i^{(j)}) \in \mathbb{R}^{B_q\times B_k} \notag\end{align}\\
    \begin{align} \text{update}\; l_i^{(j)} &= \delta_{m_i^{(j)}}l_i^{(j-1)} + \mathrm{rowsum}(S_i^{(j)})\in \mathbb{R}^{B_q}  \notag\end{align}\\
    \begin{align} \text{update}\; \tilde{O_i}^{(j)} &= \mathrm{diag}(\delta_{m_i^{(j)}})^{-1}\tilde{O_i}^{(j-1)} + S_i^{(j)}V_j\in \mathbb{R}^{B_q\times d} \notag\end{align}\\
    \end{cases}
    $$

    where $\delta_{m_i^{(j)}} := \exp(m_i^{(j-1)}\!-m_i^{(j)})$



## Backward

*you can find the tutorial of how to implement flash-attn backward kernel with triton [here](../../notebooks/tutorial_triton.ipynb)*

### For standard attn backward:


$$
\begin{cases} 
\begin{align}\mathrm{d}{V} &= A^{\mathrm T} \mathrm{d}{O} \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{A} = \mathrm{d}{O}V^{\mathrm T} \in \mathbb{R}^{N\times N} \notag\end{align} \\
\begin{align} \mathrm{d}{P}_{i:} = \cfrac{\partial A_{i:}}{\partial P_{i:}}\cdot\mathrm{d}{A}_{i:}\in \mathbb{R}^{N}, \quad
where\; \cfrac{\partial A_{i:}}{\partial P_{i:}} = J_{softmax} = \mathrm{diag}(A_{i:}) - A_{i:}A_{i:}^{\mathrm T} \in \mathbb{R}^{N\times N} \notag\end{align} \\
\begin{align}\mathrm{d}{Q} &= \mathrm{d}{P}K \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{K} = \mathrm{d}{P}^{\mathrm T}Q \in \mathbb{R}^{N\times d} \notag\end{align}
\end{cases}
$$

given $\mathrm{d}{O} \in \mathbb{R}^{N\times d}$, where $\mathrm{d}X$ denotes $\cfrac{\partial{\mathbb{loss}}}{\partial{X}}$, and $X_{i:}$ gets the column vector made of the $i$-th row of $X$, for any matrix $X$


### For flash-attn backward:


#### step0. store LSE during forward to save memory:

$$\text{for i-th row}: \begin{cases}
\begin{align} \text{since}\; A_{i:} &= \cfrac{S_{i:}}{l_{i:}} \in \mathbb{R}^{B_k}, \quad l_{i} = \mathrm{sum}(S_{i:}) \in \mathbb{R}, \quad S_{i:} = \exp(P_{i:} - m_{i}) \in \mathbb{R}^{B_k}, \quad m_{i} = \max(P_{i:})\in \mathbb{R} \notag\end{align}\\
\begin{align} \text{therefore}\; A_{i:} &= \cfrac{\exp(P_{i:} - m_{i})}{\mathrm{sum}(\exp(P_{i:} - m_{i}))} = \cfrac{\exp(P_{i:} - m_{i})}{\exp(\mathrm{LSE}(P_{i:} - m_{i}))} = \exp(P_{i:} - (m_{i} + \mathrm{LSE}(P_{i:} - m_i))) \notag\end{align}\\
\begin{align} \text{and according to}\; \text{LSE}(\bold{x}) = \max(\bold x) + \text{LSE}(\bold{x}-\max(\bold x)) \notag\end{align}\\
\begin{align} \text{therefore}\; A_{i:} &= \exp(P_{i:} - (m_{i} + \mathrm{LSE}(P_{i:} - m_i))) = \exp(P_{i:} - \mathrm{LSE}(P_{i:})) = \exp(P_{i:} - \mathrm{LSE}_i)\notag\end{align}\\
\end{cases}
$$

so we can jump storing $m_i, l_i$ to compute $S_{i:}$, but computing $A_{i:}$ from $P_{i:}$ directly with only $\mathrm{LSE}_i$


#### step1. compute Delta during preprocessing to save memory:

$$\text{for i-th row}: \begin{cases}
\begin{align} \text{since}\; \mathrm{d}{P}_{i:} &= \cfrac{\partial A_{i:}}{\partial P_{i:}}\cdot\mathrm{d}{A}_{i:} = (\mathrm{diag}(A_{i:}) - A_{i:}A_{i:}^{\mathrm T} )\cdot\mathrm{d}{A}_{i:} = A_{i:}\odot\mathrm{d}{A}_{i:} - (A_{i:}A_{i:}^{\mathrm T})\mathrm{d}{A}_{i:}  \in \mathbb{R}^{B_k}\notag\end{align}\\
\begin{align} \text{then}\; \mathrm{d}{P}_{i:} &= A_{i:}\odot\mathrm{d}{A}_{i:} - A_{i:}(A_{i:}^{\mathrm T}\mathrm{d}{A}_{i:}) = A_{i:}\odot\mathrm{d}{A}_{i:} - (A_{i:}^{\mathrm T}\mathrm{d}{A}_{i:})A_{i:}\notag\end{align}\\
\begin{align} \text{define}\; \Delta_{i} = A_{i:}^{\mathrm T}\mathrm{d}{A}_{i:}  \in \mathbb{R}, \; \text{and because}\; \mathrm{d}{A}_{i:} = (\mathrm{d}{O}_{i:}^{\mathrm T}V^{\mathrm T})^{\mathrm T} = VdO_{i:}  \in \mathbb{R}^{B_k}\notag\end{align}\\
\begin{align} \text{so}\; \Delta_{i} = A_{i:}^{\mathrm T}\mathrm{d}{A}_{i:} = A_{i:}^{\mathrm T}(VdO_{i:}) = (A_{i:}^{\mathrm T}V)dO_{i:} = O_{i:}^{\mathrm T}dO_{i:}\notag\end{align}\\
\begin{align} \text{therefore}\; \mathrm{d}{P}_{i:} = A_{i:}\odot\mathrm{d}{A}_{i:} - (A_{i:}^{\mathrm T}\mathrm{d}{A}_{i:})A_{i:} = A_{i:}\odot\mathrm{d}{A}_{i:} - \Delta_{i}A_{i:} = A_{i:}\odot (\mathrm{d}{A}_{i:} - \Delta_{i}) \notag\end{align}\\
\begin{align} \text{then for all rows, we compute }\; \Delta = \mathrm{rowsum}(O\odot dO)\in \mathbb{R}^{B_q}\; \text{during preprocessing} \notag\end{align}\\
\end{cases}
$$

so we can avoid massive matrix computing like $A_{i:}A_{i:}^{\mathrm T} \in \mathbb{R}^{B_k\times B_k}$


#### step2. flash-attn backward algorithm with recomputation (double-loop):

* the outer loop runs through $j := 1 \rightarrow N_k$ for each block of $K_j, V_j$ to compute $dK_j, dV_j$,  where $N_k = \lceil\frac{N}{B_k}\rceil$

    $$\text{in one j-th outer iteration}: \begin{cases}
    \begin{align} \text{load}\; K_j, V_j \in \mathbb{R}^{B_k\times d}\; \text{from HBM to SRAM, and initialize}\; dK_j^{(0)}, dV_j^{(0)} = (0)_{B_c\times d} \in \mathbb{R}^{B_k\times d} \notag\end{align} \\
    \begin{align} \text{loop over}\; i := 1 \rightarrow N_q\; \text{for each i-th inner iteration} \notag\end{align} \\
    \begin{align} \text{write}\; dK_j = dK_j^{(N_q)}, dV_j = dV_j^{(N_q)} \;\text{back to HBM to return as output} \notag\end{align} \\
    \end{cases}
    $$


* in which each inner loop goes across $i := 1 \rightarrow N_q$ for each block of $Q_i, O_i, dO_i$ to update $dQ_i, dK_j^{(i)}, dV_j^{(i)}$, where $N_q = \lceil\frac{N}{B_q}\rceil$

    $$\text{in one i-th inner iteration}: \begin{cases} 
    \begin{align} \text{load}\; Q_i, O_i, dO_i, \mathrm{LSE}_i, \Delta_i\; \text{from HBM to SRAM} \notag\end{align} \\
    \begin{align} \text{recompute}\; P_j^{(i)} &= Q_iK_j^{\mathrm T} \in \mathbb{R}^{B_q\times B_k} \notag\end{align} \\
    \begin{align} \text{recompute}\; A_j^{(i)} &= \exp(P_j^{(i)}\!- \mathrm{LSE}_i) \in \mathbb{R}^{B_q\times B_k} \notag\end{align} \\
    \begin{align} \text{update}\; dV_j^{(i)} &= dV_j^{(i-1)} + (A_j^{(i)})^{\mathrm T} dO_i \in \mathbb{R}^{B_k\times d} \notag\end{align} \\
    \begin{align} \text{compute}\; dA_j^{(i)} &= dO_iV_j^{\mathrm T} \in \mathbb{R}^{B_q\times B_k} \notag\end{align} \\
    \begin{align} \text{compute}\; dP_j^{(i)} &= A_j^{(i)}\odot (dA_j^{(i)} - \Delta_i) \in \mathbb{R}^{B_q\times B_k} \notag\end{align} \\
    \begin{align} \text{update}\; dK_j^{(i)} &= dK_j^{(i-1)} + (dP_j^{(i)})^{\mathrm T} Q_i \in \mathbb{R}^{B_k\times d} \notag\end{align} \\
    \begin{align} \text{load}\; dQ_i \;\text{from HBM to SRAM, then update}\; dQ_i \leftarrow dQ_i + dP_j^{(i)}K_j \in \mathbb{R}^{B_q\times d},\; \text{write it back to HBM} \notag\end{align} \\
    \end{cases}
    $$