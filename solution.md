We have this in our derivation:

$$
r(c) = \frac{1 + \text{sign}\left((W_u^\top \phi(c)) \cdot (W_\ell^\top \phi(c)) + b\right)}{2}
$$

But the question wants this form:

$$
r(c) = \frac{1 + \text{sign}\left(\tilde{W}^\top \tilde{\phi}(c) + \tilde{b}\right)}{2}
$$

So the task is:

> **Convert** the scalar expression $(W_u^\top \phi(c)) \cdot (W_\ell^\top \phi(c))$ into a **linear model** in terms of a new feature map $\tilde{\phi}(c)$, such that it looks like a dot product with $\tilde{W}$.

Let’s go step by step.

---

### 🔁 Step 1: Use bilinear form

We are starting from:

$$
(W_u^\top \phi(c)) \cdot (W_\ell^\top \phi(c)) = \left(W_u^\top \phi(c)\right)\left(W_\ell^\top \phi(c)\right)
$$

This is a scalar. Let's write it as a matrix product:

$$
= \left(W_u^\top \phi(c)\right)\left(W_\ell^\top \phi(c)\right) = (\phi(c)^\top W_u)(W_\ell^\top \phi(c))
$$

This can be rearranged as:

$$
= \phi(c)^\top W_u W_\ell^\top \phi(c)
$$

Let:

$$
M := W_u W_\ell^\top \in \mathbb{R}^{D \times D}
$$

So:

$$
(W_u^\top \phi(c)) \cdot (W_\ell^\top \phi(c)) = \phi(c)^\top M \phi(c)
$$

This is a **quadratic form**. But we want a **linear model** in terms of a **new feature map**.

---

### 🧠 Step 2: Use identity to flatten the quadratic form

This identity is key:

> For any vector $\phi \in \mathbb{R}^D$ and matrix $M \in \mathbb{R}^{D \times D}$,

$$
\phi^\top M \phi = \text{vec}(M)^\top (\phi \otimes \phi)
$$

#### Proof:

Let’s break it down. Let:

* $\phi(c) \in \mathbb{R}^D$
* $M \in \mathbb{R}^{D \times D}$

Then:

$$
\phi^\top M \phi = \sum_{i=1}^{D} \sum_{j=1}^{D} M_{ij} \phi_i \phi_j
$$

Now consider:

$$
(\phi \otimes \phi) \in \mathbb{R}^{D^2}
$$

whose elements are all $\phi_i \phi_j$ for all combinations of $i,j$.

And:

$$
\text{vec}(M) \in \mathbb{R}^{D^2}
$$

is the matrix $M$ flattened column-wise.

Then:

$$
\text{vec}(M)^\top (\phi \otimes \phi) = \sum_{i,j} M_{ij} \phi_i \phi_j = \phi^\top M \phi
$$

So it works.

---

### ✅ Step 3: Define the new feature map and weight vector

Let:

* $\tilde{\phi}(c) := \phi(c) \otimes \phi(c) \in \mathbb{R}^{D^2}$
* $\tilde{W} := \text{vec}(W_u W_\ell^\top) \in \mathbb{R}^{D^2}$

Then:

$$
(W_u^\top \phi(c)) \cdot (W_\ell^\top \phi(c)) = \phi(c)^\top (W_u W_\ell^\top) \phi(c) = \tilde{W}^\top \tilde{\phi}(c)
$$

---

### 🎯 Final Form

So we can write:

$$
r(c) = \frac{1 + \text{sign}\left(\tilde{W}^\top \tilde{\phi}(c) + \tilde{b}\right)}{2}
$$

Where:

* $\tilde{\phi}(c) := \phi(c) \otimes \phi(c)$
* $\tilde{W} := \text{vec}(W_u W_\ell^\top)$
* $\tilde{b} := b$

This is now in the exact format that the question demands — a **single linear model** over a derived feature map $\tilde{\phi}(c)$ that depends **only on the challenge**, not on PUF-specific parameters.

---

### 📌 Notes

* This works because bilinear/quadratic expressions can always be written as **linear functions** over the outer product features.
* We must **vectorize** the tensor product if we want the weight vector $\tilde{W}$ to be in $\mathbb{R}^{D^2}$
* This technique is used in machine learning kernels, e.g. polynomial kernels.
