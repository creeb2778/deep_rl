# deep_rl
deep reinforcement learning


# System Dependencies

python 3.6+
torch 0.4.1
unityagents 0.4.0
 
# Project Details

The goal of this project is to train an agent to navigate a world filled with bananas. There are yellow and blue bananas, but we only want our agent to collect the yellow ones! This is based off the unity banana simulation, leveraging the 'bananabrain' brain. 

The observation state space is a vector containing 37 values. At each step, the agent can choose 4 possible actions: move up, down. left, or right. The agent is given a reward of +1 for every yellow banana it collects, and a reward of -1 for each blue banana it incorrectly collects. We do not allow the agent to take more than 1K actions per episode in this task. 

The environment is considered solved when the agent's average score for the past 100 episodes exceeds 13. 

# Running the Code
The code consists of an agent (doub_dqn_agent.py), a neural network (banana_mod1.py), and code to train the agent given the banana environment (ddqn_agent_training.ipynb). 

The neural net is imported in the agent script, so you only need to import the agent from doub_dqn_agent.py to get started. There is a function called DDQN in the training file that steps through the environment and passes the correct state, action, next state, reward tuple to the agent for training. I leaved the results and some visualizations in the training file (as well as environment specs) to give concrete examples of how to interact with the scripts. 

The model weights are saved for every 100 trial group that solved the environment. You can import the 1800checkpoint.pth to view a late stage result. All the others can be found in the folder other_weights, or provided upon request. 
