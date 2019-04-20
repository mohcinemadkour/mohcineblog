Title: Deep Reinforcement Learning
Date: 2018-06-14 13:01
Category: Deep Learning
Tags: Deep Reinforcement Learning
Slug: Deep Reinforcement Learning
Author: Mohcine Madkour
Illustration: data-lake-background.png
Email:mohcine.madkour@gmail.com

Today, we will explore Reinforcement Learning – a goal-oriented learning based on interaction with environment. Reinforcement Learning is said to be the hope of true artificial intelligence. And it is rightly said so, because the potential that Reinforcement Learning possesses is immense.
Reinforcement learning refers to goal-oriented algorithms, which learn how to attain a complex objective (goal) or maximize along a particular dimension over many steps; for example, maximize the points won in a game over many moves. They can start from a blank slate, and under the right conditions they achieve superhuman performance. Like a child incentivized by spankings and candy, these algorithms are penalized when they make the wrong decisions and rewarded when they make the right ones – this is reinforcement.

#Introduction to reinforcement learning
## The learning paradigm

The RL is kind of learning by doing, with no supervisor, but only a reward signal. The Feedback is  delayed and not instantaneous. In this kind of learning the time really matters and the agent’s actions affect the subsequent data it receives
A reward Rt is a scalar feedback signal, It indicates how well agent is doing at step t. The agent’s job is to maximise cumulative reward. Reinforcement learning is based on the reward hypothesis, which states that all goals can be described by the **Maximisation of expected cumulative reward**. 
Examples of Rewards can be +ve reward for following desired trajectory and −ve reward for crashing.
The goal is to select actions to maximise total future reward. The actions may have long term consequences, and the reward may be delayed. Sometimes It may be better to sacrifice immediate reward to gain more long-term reward
Examples are numerous. For example a financial investment (may take months to mature), and Refuelling a helicopter (might prevent a crash in several hours)
Two fundamental problems in sequential decision making 
![supervised vs rl](/images/dsrl.png)
## Sequential Decision Making
The reinforcement learning id a Sequential Decision Making process. In general there is two types of environmnets: Fully Observable Environments which is recommended for Markov decision process in where the Agent state about the environmnet is identical with the environment state and with the information state; and the Partially Observable Environments in which the Partially Markov decision process can be applied. In this environment, the agent indirectly observes environment. The Agent must construct its own state representation whcih includes complete history, beliefs of environment state. The Recurrent neural network can be used in this case.
##Components of an RL Agent
An RL agent may include one or more of these components: Policy: agent’s behaviour function, Value function: how good is each state and/or action, and Model: agent’s representation of the environment. The **Policy** is the agent’s behaviour. It is a map from state to action, e.g. We have two types : Deterministic policy: a = π(s) and Stochastic policy: π(a|s) = P[At = a|St = s]. The **Value Function** is a prediction of future reward. It is used to evaluate the goodness/badness of states And therefore to select between actions, e.g.]:**vπ(s)** = Eπ [Rt+1 + γRt+2 + γ2 Rt+3 + ... | St = s]. **The model** predicts what the environment will do next. The P predicts the next state, and the R predicts the next (immediate) reward.
Pss'= P[St+1 = s | St = s, At = a], Ras = E [Rt+1 |St = s, At = a]

#Types of reinforcement learning algorithms
RL algorithms that satisfy the *Markov property* are called the *Markov Decision Processes (MDP)*. The Markov property assumes that the current state is independent of the path that leads to that particular state. 
![Markovian environments and Non Markovian environments](/images/markov.png)
Hence, in Markovian problems a memoryless property of a stochastic process is assumed. In practice it means that the probability distribution of the future states depends only on the current state and not on the sequence of events that preceded. This is a useful property for stochastic processes as it allows for analysing the future by setting the present
![State Transition from state s to state s'](/images/fig1_rl.png)
An MDPs consist of state (s), action (a) sets and given any state and action to be taken, a transition probability function of each possible next state (s’) illustrated in figure 1. In addition, each taken action to arrive to the next state is rewarded giving each of all possible actions a reward value
depending on the type of action. Each visited state is accredited by a value given to it according to a **value function V(s)** which represents how good it is for an agent to be in a given state. The value of a state s under a policy π is then denoted as Vπ(s) which in theory denotes the expected return when starting in state s and following a sequence of states to be visited according to the order defined in π thereafter. When this theorem is applied to a model-free control problem, the **state-value function** may
not suffice as it does not show what action was taken for the state value to be acquired. Therefore, a similar function has been introduced representing an estimation of the value of each possible action in a state. This is described as the **action-value function** for policy π Qπ(s,a). Figure 2 illustrates an example of the relationship between the action-value function and the state-value function. In 2.a. the action-values are shown for each direction of the propagation, North, East, South, and West respectively. **The state value function represents then the highest action-value possible in that state which is the action North in the example**.![1a 1b](/images/rl_fig2.png)

The optimal policy is denoted as the superscript asterisk to the action-value-function Q(s,a) and state value-function V(s). Formally, the optimal value function is then given by:
![Eq1](/images/eq1.png)
Where Q*(s,a) is given by:
![Eq2](/images/eq2.png)
Herein, T(s, a, s’) is the transition probability to the next state s’ given state s and action a. γ presents the discount factor which is usually smaller than 1 and is used to discount for earlier values in order to assign
higher values for sooner rewards. This is necessary to converge the algorithm.
Substituting equation 3 in 2 gives the Bellman equation:
![Eq3](/images/eq3.png)
These updates will be appended to the states that were visited resulting (after a significant number of iterations) in state values showing how good to be in that state. In order to be able to choose between the states to select a policy, **as many states as possible need to be visited** in order to converge to an accurate estimation of the state value. Acquiring the highest reward depends on these visited states and the reward accumulated. However, in order to discover more states and potentially higher rewards, the agent needs to take actions it has never taken before. This is referred to as the **trade-off between exploitation and exploration**. This trade-off could be achieved by setting a variable denoted as Epsilon (ε) which gives the extent of exploration versus exploitation. A fully exploiting policy is referred to as an
epsilon-greedy policy and holds a value of 0 for ε. Correspondingly, a fully exploring policy gives a value of 1 to ε and is referred to as an epsilon-soft policy. The learning can therefore be tuned between these two extremes in order to allow for convergence towards an optimal value by occasionally exploring new states and actions.

##Categorisies of RL agents
Reinforcement learning is like trial-and-error learning. The agent should discover a good policy from its experiences of the environment and Without losing too much reward along the way. The **Exploration** finds more information about the environment. The **Exploitation** exploits known information to maximise reward. It is usually important to explore as well as exploit.
An agent can evaluate the future Given a policy (**Prediction**) or optimise the future and find the best policy (**Control**)
There is five types of agents: **Value Based** No Policy (Implicit)+ Value Function, **Policy Based**: Policy + No Value Function, **Actor Critic**: Policy+ Value Function, **Model Free**: Policy and/or Value Function+ No Model
, **Model Based**: Policy and/or Value Function+ Model
![RL Agents](/images/RLAgents.png)

##Classes of RL algorithms
RL knows three fundamental classes of methods for solving these learning problems: **Dynamic Programming (DP)**, **Monte Carlo methods**,  **Temporal-difference learning**
Dependent on the problem at stake, each of these methods could be more suitable than the other. **DP** methods are model-based and require therefore a complete and accurate model of the environment i.e. all the aforementioned functions of the environment need to be known to initiate learning. However,
the environment is not always defined prior to the learning process which poses a challenge to this method. This is where the two other **model-free** learning methods come in handy. The **Monte Carlo** algorithms only require an experience sample such as a data set in which the states, actions and rewards
of the (simulated) interaction with the environment. In comparison with DP methods, no model of the **transition probability function** is required and neither the **dynamics** of the environment. Monte Carlo algorithms solve the RL problem by **averaging** sample return of each **episode**. Only after the termination of an episode, that the value **estimation** and **policies** are updated. Hence, it is based on averages of complete returns of the value functions of each state. This class of algorithms does not exploit Markov property described before and is therefore more efficient in **non-Markovian** environments. On the other hand, **Temporal-Difference methods** do also not require a model of the environment but are like DP solving for incrementing **step-by-step** rather than **episode-by-episode**. Hence, TD methods exploit the **Markovian property** and perform usually better in Markovian environments.
The choice between these two classes of model-free RL algorithms very much depends on the type of data set available. For continuous processes in which there are no fixed episodic transitions, **Monte Carlo** methods may not be the optimal solution as they average the return only at the end of each episode. **TD** algorithms might then be a better solution as they assign a reward incrementally over each state. This allows them to converge faster towards an optimal policy for large data sets with a large number state spaces.

##Temporal-difference learning: On-policy and off-policy TD control
**TD** algorithms comprise two important RL classes of algorithms divided in **Off-Policy** and **On-Policy** TD control algorithm classes. The difference between the two lays in the policy that is learned from the simulation or set of experiences (data). **On-Policy TD control** algorithms are often referred to as **SARSA algorithms** in which the letters refer to the sequence of State, Action, Reward associated with the state transition, next State, next Action. This sequence is followed in each **time-step** and is used to update the **action-value** of these two states according:
![Eq4](/images/eq4.png)
Here, α represents the step-size parameter which functions as the exponentially moving average parameter. It is especially useful for **non-stationary** environments for weighting recent rewards more heavily than long-past ones. This could also be illustrated by rearranging the above equation to:
![Eq5](/images/eq5.png)
If α is a number smaller than one for non-stationary environments which indicates that recent updates weight more than previous ones. This transition happens after every nonterminal state. The Q (st+1 ,at+1 ) components of every terminal state is defined as zero. Hence, every terminal state has an update value of 0. **SARSA** is called an on-policy algorithm because it updates the **action-value-function** according to the **policy** it is taking in every **step**. Therefore, it takes the epsilon-policy into account in order to arrive the optimal policy for a certain problem. **Off-policy** algorithms approximate the best possible policy even when that policy is not taken by the agent. Hence, **Off-Policy** algorithms base the update of the **state action-value** function on the assumption of **optimal behaviour** without taking into account the **epsilon policy** (the chance to take a negative action). The cliff figure shows a suitable example given by Sutton and Barto (1998) and which illustrates the policy outcome differences between the two types of TD algorithms ![Cliff)](/images/cliff.png). The cliff represents states with high negative reward. Since **SARSA** takes the **epsilon policy** into account, it learns that at some instances a non-optimal action will be taken which results in a high negative reward. Hence, it will learn to take the safe path rather than the optimal path. **Q-learning algorithms** on the other hand, will take the optimal path by which the highest total reward could be achieved. This is because it does not take the **epsilon probability** into account of taking an extremely negative action. This class of algorithms is denoted by the following equation:
![Eq6](/images/eq6.png)
This difference will inevitably influence the suitability for the type of application. 

#Markov Decision Processes
Markov decision processes formally describe an environment for reinforcement learning Where the environment is fully observable, i.e. The current state completely characterises the process.
![MDP](/images/MDP.png)
Almost all RL problems can be formalised as MDPs, e.g.Optimal control primarily deals with continuous MDPs, Partially observable problems can be converted into MDPs, Bandits are MDPs with one state
The Markov Property states that "The future is independent of the past given the present” in other ways a state St is Markov if and only if
P [S t+1 | S t ] = P [S t+1 | S 1 , ..., S t ]
The state captures all relevant information from the history and once the state is known, the history may be thrown away. i.e. The state is a sufficient statistic of the future.

For a Markov state s and successor state s' , the state transition
probability is defined by Pss' = P[St+1 = s'| St = s]. The State transition matrix P defines transition probabilities from all states s to all successor states s' ![State Transition Matrix](/images/State_Transition_Matrix.png)where each row of the matrix sums to 1

A Markov process is a memoryless random process, i.e. a sequence
of random states S1 , S2 , ... with the Markov property. Otherwise it is a tuple <S,P> with S is a (finite) set of states, P is a state transition probability matrix, Pss'= P [S t+1 = s'| St = s]
![Example](/images/markov_process.png)

A Markov reward process is a Markov chain with values.
Definition: A Markov Reward Process is a tuple <S, P, R, γ>
S is a finite set of states
P is a state transition probability matrix,
P ss'= P [St+1 = s'| St = s]
R is a reward function, Rs = E [Rt+1 | St = s]
γ is a discount factor, γ ∈ [0, 1]

The return Gt is the total discounted reward from time-step t.
G t = Rt+1 + γRt+2 + ...
The discount γ ∈ [0, 1] is the present value of future rewards
The value of receiving reward R after k + 1 time-steps is γkR.
This values immediate reward above delayed reward.
γ close to 0 leads to ”myopic” evaluation
γ close to 1 leads to ”far-sighted” evaluation

Most Markov reward and decision processes are discounted:
* Mathematically convenient to discount rewards
* Avoids infinite returns in cyclic Markov processes
* Uncertainty about the future may not be fully represented
* If the reward is financial, immediate rewards may earn more
interest than delayed rewards
* Animal/human behaviour shows preference for immediate
reward
* It is sometimes possible to use undiscounted Markov reward
processes (i.e. γ = 1), e.g. if all sequences terminate.

The value function v (s) gives the long-term value of state s
Definition : The state value function v (s) of an MRP is the expected return
starting from state s v (s) = E [G t | S t = s]
![Value Function](/images/Value_Function.png)


#Final Words
Reinforcement learning is extremely fun but hard topic. I am excited to learn more!