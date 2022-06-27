# Free Energy Principle: Simulation and Explanation

Stuart Truax, 2022-06


 This repository contains a detailed explanation and dynamical simulation of the
 __free energy principle (FEP)__. There are two components to the repo:

 1. An explanation and informal derivation on the FEP, based on [1].
 2. A dynamical simulation of a coupled mechanical-electrochemical system (i.e. the "primordial soup") described in [1].

The simulation is <a href="">here</a>.



# Free Energy Principle

The **free energy principle** (FEP) is a principle that describes how
complex systems, through interaction with their surroundings, achieve
non-equilibrium steady-states through the minimization of the number of
internal states of the system [2]. The
FEP utilizes the formalisms of random dynamical systems and Bayesian
inference to describe how a complex system can converge to a steady
state that is dependent upon feedback from the external environment. The
convergence to the steady state is an example of **variational
inference** (i.e. coverging to a probability distribution via an
optimization procedure) .

The FEP is useful in describing the formation of **self-organizing
systems**. A claimed consequence of the principle is that
self-organization is an emergent property of any ergodic random
dynamical system that possesses a Markov blanket [1].

## Background and Motivation: Variational Inference

Given some data $\mathcal{D}$, there often arises the task of estimating
some posterior distribution $p(\mathbf{x} | \mathcal{D})$, where
$\mathbf{x}$ is some set of variables, e.g. a state vector. Often the
calculation of
$p(\mathbf{x}, \mathbf{\theta})  \equiv p(\mathbf{x} | \mathcal{D})$ is
intractable in itself, and must be approximated by a series of
converging estimates from some tractable family of probability
distributions. This is the method of **variational inference**
[3].

Assume the following definitions:

-   $p(\mathbf{x}, \mathbf{\theta})$ - The \"true\" yet intractable
    probability distribution, with random variable $\mathbf{x}$ and
    parameters $\mathbf{\theta}$.

-   $q(\mathbf{x}, \mathbf{\theta})$ - The variational estimate of
    $p(\mathbf{x}, \mathbf{\theta})$.

The **Kullback-Leibler (KL) divergence** provides an immediate way of
evaluating the distance between the estimate $q$ and the true
distribution $p$:

$$KL(p||q) = \sum_{\mathbf{x}} p(\mathbf{x}) \text{log} \left ( \frac{p(\mathbf{x}) }{q(\mathbf{x}) } \right )$$

Evaluating this form of the KL divergence involves calculating the log
expectation of $p(\mathbf{x})$, which is intractable. An alternative
approach is to use the reverse ordering of the arguments, which is not
equivalent to the above definition (due to the asymmetry of the KL
divergence), but is more tractable:

$$KL(q||p) = \sum_{\mathbf{x}} q(\mathbf{x}) \text{log} \left ( \frac{q(\mathbf{x}) }{p(\mathbf{x}) } \right )$$

In variational inference, one intentionally chooses $q(\mathbf{x})$ to
be from a tractable family of distributions. Therefore, the
log-expectation of $q(\mathbf{x})$ in this second equation is tractable.
However, the denominator in the $\text{log}()$ term requires some extra
work. Recall that
$p(\mathbf{x}, \mathbf{\theta})  \equiv p(\mathbf{x} | \mathcal{D})$,
which by Bayes' Rule yields:

$$p(\mathbf{x} | \mathcal{D}) = \frac{ p(\mathcal{D}| \mathbf{x}) p(\mathbf{x})}{p(\mathcal{D})}$$

The denominator $Z = p(\mathcal{D})$ is referred to as the **partition
function** (an allusion to the same quantity in statistical mechanics).
It is calculated by the (often intractable) operation [4]
[^1] :

$$Z(\theta) = \int_{\mathbf{x}} p(\mathbf{x}) d\mathbf{x}$$

A workaround for this is to instead use the quantity:

$$\tilde{p} = p(\mathbf{x}, \mathcal{D}) = p(\mathbf{x}) p(\mathcal{D}) = p(\mathbf{x}) Z$$

Using $\tilde{p}$, the KL divergence can be restated as below. We use
this restated and tractable KL-divergence to define the loss function
$J(q)$ :

$$J(q) =   KL(q||\tilde{p}) = \sum_{\mathbf{x}} q(\mathbf{x}) log \left ( \frac{q(\mathbf{x}) }{p(\mathbf{x}) Z} \right )  = KL(q||p) - log(Z)$$

The ultimate goal in variational inference is to find the probability
distribution $q(\mathbf{x}, \mathbf{\theta})$ that minimizes $J(q)$.

[^1]: In statistical mechanics, the partition function $Z$ is often
    given in terms of the system Hamiltonian $H$, which for many systems
    can be derived analytically and often lends itself to calculable
    integrals. In the context of inference, the partition function
    instead requires integrating over all possible hypotheses
    $\mathbf{x}$ consistent with $\mathcal{D}$, often making the
    integral intractable.

### Interpretation of $J(q)$ as a Variational Free Energy

$J(q)$ can also be expressed as:

$$J(q) = E_{q}[\text{log}(q(\mathbf{x})] +  E_{q}[-\text{log}(\tilde{p}(\mathbf{x})]  = -H(q) + E_{q}[E(\mathbf{x})]$$

where $E(\mathbf{x}) = -\text{log}(\mathbf{x})$ is the energy of state
$\mathbf{x}$, $E_{q}[E(\mathbf{x})]$ is the **expected energy** of the
system, and $H(q)$ is the **entropy** of the system. In this sense,
$J(q)$ is seen as a **variational free energy**, in an analogy to the
Gibbs free energy from physics. As in physics, the variational free
energy $J(q)$ is *minimized* through the *maximization* of entropy (i.e.
the variational distribution $q(\mathbf{x})$ is maximized in terms of
entropy).



## Free Energy Principle: Theoretical Background

This section will introduce some of the theoretical concepts and
structures necessary to understand the FEP and its governing dynamics.
The exposition in this and the following section closely follows the
exposition and notation found in [1], and uses its
terminology.

We begin with a lemma [1], which is a bit informal with its
notion of \"structural and dynamical integrity\", but it broadly states
what the FEP seeks to demonstrate:

**Lemma**: Any ergodic random dynamical system that possesses a Markov
blanket will appear to actively maintain its structural and dynamical
integrity.

To add more detail to this statement, some informal definitions of
**ergodic systems** and **Markov blankets** follow:

1.  **Ergodic system**: A dynamical system for which the time average of
    a measurable function of the system converges *almost surely* over
    some finite time T.

    Call $p(\mathbf{x})$ the **ergodic density** of state $\mathbf{x}$,
    which is probability that the ergodic system is in state
    $\mathbf{x}$. An ergodic system has the following property:

    $$ p(\mathbf{x}) = \text{average proportion of time spent in state } \mathbf{x}$$

2.  **Markov blanket**: A Markov blanket of a random variable $Y$ in a
    set of random variables $S = \{X_0, ..., X_1\}$ is a set
    $S' \subseteq S$ such that $S'$ contains at least all the
    information needed to infer $Y$ (i.e. the remaining random variables
    in $S$ are redundant for the purpose of inference of $Y$)
    [5] . In the context of a Bayesian network,
    the Markov blanket of a node $Y$ consists of:

    1.  The parents of $Y$.

    2.  $Y$ itself.

    3.  The children of $Y$.

    4.  The other parents of the children of $Y$.

With these informal definitions in hand, one can define an ergodic
random dynamical system $m$ that contains a Markov blanket. Although
ergodicity and the existence of a Markov blanket are the necessary
conditions for the FEP to attain, one must add some further structure to
the system to illustrate the full results of the FEP. In
[1], this is referred to as a \"self-organizing
architecture\". The particular definition of these states in this
architecture, and the coupling between them, is essential in producing
the results of the FEP.

Let the state space of the system be partitioned into **external**,
**sensory**, **active**, and **internal** states. These states and their
dependencies are defined below. We shall refer to an \"agent\" as the
portion of the system constituted by the internal states and its Markov
blanket.

-   $\Omega$ *Fluctation Sample Space* A sample space from which random
    fluctuations are drawn (similar to the fluctuations of Boltzmann's
    microcanonical ensemble).

-   $\Psi : \Psi \times A \times \Omega \rightarrow \mathbb{R}$
    *External (Hidden) States* (i.e. of the world \"outside the
    banket\") that cause the sensory states of the agent and depend on
    actions by the agent.

-   $S: \Psi \times A \times \Omega \rightarrow \mathbb{R}$ *Sensory
    States*, which are the agent's sensations, and constitute a
    probabilistic mapping from action and external states.

-   $A: S \times \Lambda \times \Omega \rightarrow \mathbb{R}$ *Action*,
    an agent's action that depends on its sensory and internal states.

-   $\Lambda : \Lambda \times S  \times \Omega \rightarrow \mathbb{R}$
    -*Internal States* (i.e. the states of the agent), which cause
    action and depend on sensory states $S$.

-   $p(\Psi,s,a, \lambda | m)$ - *Generative density*, a probability
    over external states $\psi \in \Psi$, sensory states $s \in S$,
    active states $a \in A$ and external states $\lambda \in \Lambda$
    for a system $m$ .

-   $q(\psi | \lambda)$ - *Variational density* , an arbitrary
    probability probability density over external states $\psi \in \Psi$
    that is parameterized by internal states $\lambda \in \Lambda$.

The Markov blanket that interests us is the one defined with respect to
the *internal states* $\Lambda$, for it is the self-organization of
these internal states that is the key result of the FEP. Note that the
Markov blanket with respect to $\Lambda$ is implicit in the definition
of the states and their dependencies. If one chose different definitions
of the states and their dependencies, the boundary of the Markov
blanket, and therefore our definition of \"agent\", would change.

The coupling between the states of this system can be illustrated in the
causality graph of Figure [1](#fig:markov_blanket). The blue oval in
Figure [1](#fig:markov_blanket) denotes the Markov blanket with respect
to the internal states (i.e. the \"agent\").



![__Figure 1.__ _The casuality graph for the FEP. The state variables and their
dynamics are represented by the nodes. The couplings between nodes are
represented by the arrows. The Markov blanket of the internal state
$\lambda$ is represented by the blue oval (i.e. it encompasses the
sensory, active, and internal states)\label{markov_blanket}](images/causality_graph_w_markov_blankets.png)

__Figure 1.__ _The casuality graph for the FEP. The state variables and their
dynamics are represented by the nodes. The couplings between nodes are
represented by the arrows. The Markov blanket of the internal state
$\lambda$ is represented by the blue oval (i.e. it encompasses the
sensory, active, and internal states)._

The dynamics of the interaction between these states will be described
in next section.

## Free Energy Principle: Dynamics

With the definition of the ergodic dynamical system $m$ and its states
in hand, this section is spent deriving some mathematical results that
describe the general the dynamics of $m$. The ergodicity of the system
and the existence of the Markov blanket will ultimately yield what
Friston calls \"a fundamental and ubiquitious causal architecture for
self organization\" [1], which is no small claim.

Let the dynamics of the system be generally defined as follows:

$$\dot{x} = \underbrace{f(x)}_\text{a flow}   +  \underbrace{\omega}_\text{random fluctuation} \tag{1}$$

wherein the flow can be decomposed into the following:

$$f(x) = \begin{bmatrix}
f_{\psi}(\psi,s,a) \\
f_{s}(\psi,s,a) \\
f_{a}(s,a,\lambda) \\
f_{\lambda}(s,a,\lambda)
\end{bmatrix}  \tag{2} \label{eq:dynamics_vector}$$

with each function $f$ describing the dynamics for the respective
state variable.

As stated previously, the system is ergodic, which implies that it will
eventually converge to a **random global attractor**. Define the ergodic
density $p(x|m)$. Since this system is perturbed by random fluctuations
$\omega$ in combination with the deterministic flow $f(x)$, the dynamics
of $p(x|m)$ are governed by the **Fokker-Plank equation** (also known as
the Kolmogorov forward equation):

$$\dot{p}(x|m) = \nabla \cdot \Gamma \nabla p - \nabla \cdot (fp) \tag{3} \label{eq:fp}$$

where $\Gamma$ is the diffusion tensor. Given a covariance matrix
$\Sigma_{\omega}$ of the fluctuations $\omega$, the diffusion tensor is
defined by $\Gamma  = \frac{1}{2} \Sigma_{\omega}$.

Since $f$ is a flow, it follows that the Fundamental Theorem of Vector
Calculus (a.k.a. the Helmholtz decomposition) can be applied to $f$ to
decompose it into conservative (i.e. curl-free) and solenoidal (i.e.
divergence-free) fields:

$$f = - (\Gamma + R) \cdot \nabla G
\tag{4} \label{eq:helmholtz}$$  

where

-   $R(x)$ is an asymmetric matrix such that $R(x) = - R(x)^{T}$

-   $G(x)$ is a scalar potential called the \"Gibbs energy\"

Inserting (4) into (3) and using
some vector calculus identities, one finds that
$p(x|m) = \text{exp}(-G(x))$ satisfies the equilibrium condition
$\dot{p} = 0$, and is thus an equilibrium solution to
(3).

Now the flow $f$ can now be expressed in terms of the ergodic density
$p$:

$$f = -(\Gamma + R) \cdot \nabla \text{log} (p(x|m))
\tag{5}  \label{eq:helmholtz2}$$

which is just (4) with $p(x|m) = \text{exp}(-G(x))$ inserted.
From [5] and
[2] it follows that:

$$f_{\lambda}(s,a,\lambda) = (\Gamma + R) \cdot \nabla_{\lambda} \text{log}(p(\psi,s,a,\lambda | m))
\tag{6} \label{eq:active_states_grad}$$
$$f_{a}(s,a,\lambda) = (\Gamma + R) \cdot \nabla_{a} \text{log}(p(\psi,s,a,\lambda | m))
\tag{7} \label{eq:internal_states_grad}$$

With
(6) and (7) in hand, one can observe that the
flow is a gradient ascent (i.e. a positive gradient) of the log ergodic
density $p$. In other words, the flow performs an implicit
*maximization* of the joint probability $p(\psi,s,a,\lambda | m)$ as a
function of active and internal states. It is this result that
establishes these dynamics as a framework for variational inference.
Furthermore, as pointed out in [1], the flow follows the
isocontours of $p$ in an ascending circular manner while being
influenced by the external states, which are hidden behind the Markov
blanket.

At this point, one can invoke the **ergodic theorem** to deduce that for
any point $v \in V = S \times A \times \Lambda$ (i.e. any point within
the Markov blanket of the internal states $\Lambda$), the flow through
this point is the *average flow under the posterior density over the
external states*. (NB: This introduction of an average of a posterior
density is an explicit connection to Bayesian inference.)

The application of the ergodic theorem yields the following:

$$f_{\lambda}(v) = E_{t}[\dot{\lambda}(t) \cdot [x(t) \in v]]  = \int_{\Psi} p(\psi | v) \cdot (\Gamma + R) \cdot \nabla_{\lambda} \text{log}(p(\psi,v | m)) d\Psi
\tag{8}$$
$$f_{a}(v) = E_{t}[\dot{a}(t) \cdot [x(t) \in v]]  = \int_{\Psi} p(\psi | v) \cdot  (\Gamma + R) \cdot \nabla_{a} \text{log}(p(\psi,v | m)) d\Psi
\tag{9}$$

$$\Rightarrow$$

$$f_{\lambda}(v) = (\Gamma + R) \cdot \nabla_{\lambda} \text{log}(p(v | m))
\tag{10} \label{eq:internal_states_grad_2}$$
$$f_{a}(v) = (\Gamma + R) \cdot \nabla_{a} \text{log}(p(v | m))
\tag{11} \label{eq:active_states_grad_2}$$

In essence,
(10) and
(11) have shown that the time average of
the internal and active states passing through a point $v$ is obtained
by taking an expectation of the dynamics over the posterior probability
density of external states $p(\psi | v)$.

Here, the structure $[x(t) \in v]$ is called an **Iverson bracket**,
which is an indicator function which equals 1 when trajectory $x(t)$
passes through $v$, and 0 otherwise. This allows the expectation to be
taken over only those trajectories that pass through $v$.

The important result of
(10) and
(11) is that the expectation over a
posterior probability $p(\psi, v | m)$ over *external states* (which are
hidden from the internal states), has yielded the evolution of the
ergodic density for a point $v$ within the Markov blanket of the
internal states. More precisely, the $p(v | m)$ for a given point $v$
within the Markov blanket is evolving due to posterior *beliefs* about
external states.

For subsequent derivations, Friston introduces a density to quantify
these posterior beliefs :

**Def.**: Let $q(\psi | \lambda)$ be a density over external states
parameterized by internal states $\lambda$. Call this the variational
density or \"belief\".

At this point, the Bayesian nature of the variational density
$q(\psi | \lambda)$ and $p(\psi, v | m)$ is clear. It will be shown that
the dynamics of the system $m$ implicitly perform an optimization
procedure on $q(\psi | \lambda)$ through the gradient ascent. In this
manner, the dynamical system $m$ implicitly performs Bayesian
variational inference.

### Properties of the Free Energy $F$

Up to this point, the \"Gibbs energy' $G(x)$ has not been specified, and
can thus act as a \"free parameter\" to adapt the general dynamical
equations of the previous section to a variety of systems. Just as in
other differential equations within physics, $G(x)$ is simply a scalar
potential that influences how the trajectories behave within the
dynamical system. We will now define a further quantity, the free energy
$F$, which is directly equivalent, term for term, to the variational
free energy $J(q)$ of variational inference.

**Lemma**: For any Gibbs energy
$G(\psi,s,a,\lambda) = - \text{log}(p(\psi,s,a,\lambda))$, there is a
**free energy** $F(s,a,\lambda)$ that describes the flow of internal and
active states:

$$f_{\lambda}(s,a,\lambda) = -(\Gamma + R) \cdot \nabla_{\lambda} F
\tag{12} \label{eq:internal_states_free}$$

$$f_{a}(s,a,\lambda) = -(\Gamma + R) \cdot \nabla_{a} F
\tag{13} \label{eq:active_states_free}$$

and furthermore:

$$\boxed{
F(s,a,\lambda) = -\int_{\Psi} q(\psi | \lambda) ln \left (\frac{p(\psi,s,a,\lambda | m}{q(\psi | \lambda)} \right )  d\Psi = E_{q}[G(\psi,s,a,\lambda)] - H[q(\psi | \lambda)]
\tag{14} \label{eq:free_energy}
}$$

Notice that in
(13) and
(12), $F$ has taken the place of
$\text{log}(p(\psi,s,a,\lambda | m))$ in
(6) and
(7), with a change in sign and no
$\psi$ dependence, which means it should be seen as a log density over
sensory, active, and internal states.

The equality in
(14) is equivalent to the variational free energy
$J(q)$. The minimization of the free energy $F$ corresponds to
maximizing the entropy of the variational density $q(\psi | \lambda)$
(i.e. the posterior beliefs).

The proof of
(14) relies on breaking down the integral using
Bayes' Rule to obtain:

$$F(s,a,\lambda) = -\text{log}(p(\psi,s,a,\lambda | m) + D_{KL}[q(\psi | \lambda) || p(\psi | s,a,\lambda)]
\tag{15}$$

where $D_{KL}$ is the Kullback--Leibler divergence.
(15) can then be inserted into
(12) and
(13). The gradients of the $D_{KL}$ terms
in these new equations can then be shown to be zero, yielding the
dynamics of
(6) and
(7).

To conclude, due to the ergodicity of the system, the free energy $F$ is
also bounded in the following manner:

$$F(s,a,\lambda) \geq -\text{log}(p(s,a,\lambda | m))
\tag{16}$$ which further implies:

$$E_{t}[F(s,a,\lambda)] \geq E_{t}[-\text{log}(p(s,a,\lambda | m)] = H[p(s,a,\lambda | m)]
\tag{17} \label{eq:fe_bounds}$$

That is, the time average of the free energy $F$ is bounded below by the
entropy (complexity) of the ergodic density of the internal states'
Markov blanket. Conversely, the entropy of the internal states is
bounded above by the free energy $F$, which is minimized by the action
of the flow. That is to say that the complexity of the internal states
is limited by be behavior of the system. One way of interpreting this is
that action will limit the information content of an agent's beliefs.
Alternatively, Friston sees it as preserving the structural integrity of
the agent: \" \... then action places and upper bound on (the internal
states) dispersion (entropy) and will appear to conserve their
structural and dynamical integrity\"[1].

### Interpretation of the Free Energy $F$ in a Variational Inference Context

Interpreting the individual terms of
(14) in a variational inference context would
yield:

$$F(s,a, \lambda) = \underbrace{E_{q}[G(\psi,s,a,\lambda)]}_\text{Energy (accuracy of model)}  -   \underbrace{H[q(\psi | \lambda)]}_\text{Entropy (complexity of model)}
\tag{16}\label{eq:fe_expanded}$$

The first term (i.e. the \"accuracy\") of the equation is the cross
entropy of the distributions $p$ and $q$, which quantify the amount of
information one needs under $p$ to code an event from $q$
[6]. Minimizing this quantity therefore
corresponds to a more accurate generative model $p$ (i.e. outlier events
from $q$ are better accounted for by $p$).

The second term quantifies the amount of complexity (i.e. information)
in the external states can be captured by an internal state. This
quantity is sought to be maximized via the **maximum entropy (MaxEnt)
principle**[7].

A further manipulation of
(16) puts it into a more usable form. Expanding
the first term yields:

$$E_{q}[-\text{log} (p(\psi,s,a  | m))] - H[q(\psi|\lambda)]$$

$$= H[q(\psi|\lambda)]  +   D_{KL}[q(\psi | m)||(p(\psi,s,a  | m))] - H[q(\psi|\lambda)]$$

$$=  D_{KL}[q(\psi | \lambda)||(p(\psi,s,a  | m))]
\tag{17}\label{eq:KL_expanded}$$

which is the Kulbeck-Leibler divergence between $p$ and $q$. Expressed
in this way, the FEP simply seeks to minimize the difference between the
generational and variational densities.

Expanding further, and using the chain rule of probability,
(17) can be broken down into two more terms:

$$D_{KL}[q(\psi | \lambda)||(p(\psi,s,a  | m))]$$

$$= - \int q(\psi | \lambda) \text{log} \left ( \frac{q(\psi | \lambda)}{p(\psi,s,a | m)} \right ) d \psi$$

$$= - \int q(\psi | \lambda) \text{log} \left ( \frac{q(\psi | \lambda)}{p(\psi | s, a, m)} \right ) d \psi - \text{log}  (p(s | m))\int d \psi$$

$$= \underbrace{-\text{log}  (p(s | m))}_\text{surprise} + \underbrace{D_{KL} [ q(\psi | \lambda) ||  p(\psi | s,a, m) ]}_\text{divergence}
 \tag{18}$$

The first term (surprise) quantifies the degree to which a sensory state
$s$ is not predicted by $p$. The second term accounts for the divergence
between the generative and variational densities conditioned on the
sensory states. That is, the divergence term quantifies the difference
between the generative model and the hidden model of the environment.

The final statement of the principle is that:

$$\boxed{ \underbrace{-\text{log}  (p(s,a | m))}_\text{surprise} + \underbrace{D_{KL} [ q(\psi | \lambda) ||  p(\psi | s, a, m) ]}_\text{divergence}  \geq \underbrace{-\text{log}  (p(s,a | m))}_\text{surprise}}
 \tag{19}$$

## Results of the FEP

Some informal and general results of the FEP follow:

For an ergodic dynamical system $m$ possessing a Markov blanket on
internal states $\Lambda$:

-   The entropy over $p(\lambda)$ (i.e. complexity of the internal
    states) will be maximized, but limited by the action of the flow
    $f$. That is, an agent's internal states will not become
    disproportionately complex relative to its surroundings.

-   The surprise of an agents is limited by distance between an agent's
    beliefs $q(\psi|\lambda)$ and \"reality\" $p(\psi)$.

### Friston's Conclusions from the FEP about the Properties of Biological Systems

Friston concludes that biological systems contain the following
universal properties as a result of the FEP[1]:

-   Ergodicity

-   Possession of a Markov Blanket

-   Engagement in Active Inference

-   Autopoiesis (i.e. the maintenance of structural integrity through
    the creation and regeneration of oneself)

## References


[1] K. Friston,\"Life as we know it,\" *J. of the Royal Society Interface*,
10(86):20130475, 2013


[2] *Free Energy Principle*, Wikipedia\
`https://en.wikipedia.org/wiki/Free_energy_principle`


[3] K. P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT
Press, 2012, Section 21.2


[4] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016,
Chapters 18 and 19

[5] *Markov blanket*, Wikipedia\
`https://en.wikipedia.org/wiki/Markov_blanket`

[6] *Cross Entropy*, Wikipedia\
`https://en.wikipedia.org/wiki/Cross_entropy`

[7] *Principle of maximum entropy*, Wikipedia\
`https://en.wikipedia.org/wiki/Principle_of_maximum_entropy`
