# clinical-trial-optimization-Myanmar-RL

To help doctors and scientists in Myanmar run better clinical trials using machine learning with Python.

1. Define Project / Scope
  
2. Data Collection and Preprocessing
 
3. Defining the Reinforcement Learning Problem
 
4. Selecting Reinforcement Learning Algorithms
 
5. Implementation
 
6. Training the Reinforcement Learning Agent
 
7. Evaluation
 
8. Interpretation and Visualization
 
9. Documentation and Reporting
 
10. Asking and Answering Questions about the Project

Step 1: Introduction — Define Project / Scope

In this step, I’m going to learn what my project is about, what clinical trials are, and what I want to do to help. This helps me stay focused and make good decisions as I build my project.

1. What Is My Project About?

Introduction to My Project

I’m working on a project to help doctors and scientists in Myanmar run better clinical trials.  

Clinical trials are tests where doctors try new medicine to see if it works and if it’s safe.

I want to use something called machine learning to help. 

That means I’ll teach a computer to learn from data and make smart choices.  

One special type I’ll use is called Reinforcement Learning — it's like teaching the computer by letting it try things and learn from what happens.

What Does “Define Project / Scope” Mean?

This means I need to clearly say **what I’m doing in my project** and what I’m not doing.

So, here’s what I decided:

 I want to help improve clinical trials in Myanmar.
 
 I want to use data and machine learning to help doctors make better decisions.
 
 I will focus on diseases like malaria and dengue, because they are common in Myanmar.
  
 I will not work on other kinds of health projects, only clinical trials for now.

Learning About Clinical Trials

What Are Clinical Trials?

I learned that clinical trials are tests that help find out if new medicine works and is safe.  
Doctors try the medicine on small groups of people first. If it works well, they test it on more and more people.

What Are the Phases of Clinical Trials?

There are 4 steps (called phases) in a clinical trial:

1. I test the medicine on a small group of people to check if it’s safe.

2. I try it on more people to see if it works and check for side effects.
 
3. I test it on a big group and compare it to other treatments.
  
4. I keep checking it after it’s approved to make sure it stays safe.

What Do I Look at in a Clinical Trial?

When I study a clinical trial, I look at:

 The main goal – Is the medicine helping people feel better?
 
 Other results – Are there side effects?
 
 Safety – Is the medicine causing any harm?

What Makes Trials Hard in Myanmar?

In Myanmar, clinical trials can be harder because:

 Diseases like malaria and dengue are more common.
 
 Weather and living conditions can affect how people respond to medicine.
  
 Some people don’t trust doctors or live far from clinics and hospitals.

3. What Do I Want to Do in This Project?

My Goals

Here’s what I want to do:

 I want to help make clinical trials faster, so we get results quicker.
 
 I want to make sure the medicine is safe.
 
 I want to make sure the medicine really works and helps people feel better.

What Big Decisions Do I Need to Make?

To do a good job, I need to make smart choices, like:

 How much medicine should each person get?
 
 What kind of people should be in the trial?
 
How long should the trial last?

Why This Step Is Important

By thinking about all of this, I understand what my project is all about.  

Now, I have a clear plan. 

This will help me make smart decisions and build tools that can help real doctors and real people in Myanmar.

Step 2: I Collect and Prepare My Data

In this step, I find the data I need for my project and I get it ready so I can use it to train my machine learning model. 

If my data is messy, duplicated, or has missing parts, the computer can get confused. 

So I have to clean it first.

1. I Collect the Data

This means I look for information from past clinical trials. 

These are trials about diseases like malaria or dengue, which are common in Myanmar.

I can find this data on public websites that share free information. 

I can also find it in research papers where scientists explain what they tested. 

If allowed, I might also use real data from hospitals or clinics that have done these trials.

The data I need includes things like who was in the trial, such as their age, gender, and any health problems they had.

I also need to know what the trial was testing, like the name of the medicine, how it was given, and how much. 

Most importantly, I want to know what happened to the people in the trial.

Did they get better? 

Did they have any side effects?

This kind of information helps me understand what worked and what didn’t.

2. I Clean and Prepare the Data

This means I fix any problems in the data so it’s easy for the computer to understand and learn from.

First, I check for missing information. Sometimes, parts of the data are empty. 

If it’s something small, I might try to guess the missing part using the rest of the data.

If it’s not useful, I just remove it.

Next, I make sure everything looks the same. 

For example, if one person’s age is written as “18” and another person’s age is written as “eighteen”, I change both to just 18. 

This helps the computer understand it better.

Finally, I turn words into numbers. 

Computers don’t really understand words the way we do.

So if someone’s gender is written as “male”, I change it to 0.

If it says “female”, I change it to 1.

That way, the computer can read and work with the data more easily.

Cleaning the data is super important. 

If I don’t clean it properly, the computer might get confused and give me wrong answers.

By preparing the data well, I make sure everything is clear, correct, and ready to use.

Step 3: Defining the Reinforcement Learning Problem

In this step, I create a game for the computer to play. 

But this game is not just for fun. 

It’s a smart game where the computer learns how to run a clinical trial, like a doctor making decisions to help people.

1. I Build the Trial World

This means I make a pretend world where the clinical trial happens. 

This world has rules, and the computer can make choices inside it.

First, I decide what the computer can see. 

This is called the state. 

The computer will see things like how many people are in the trial, what medicine they are getting, and how those people are feeling.

Next, I decide what the computer can do. 

This is called the action. 

The computer can change the amount of medicine, choose who joins the trial, or decide how long the trial should last.

Then, I decide what is good and what is bad.

This is called the reward.

If the computer helps people get better, I give it points.

If the computer causes side effects or wastes money, I take away points.

2. I Let the Computer Practice

This means I let the computer play in the pretend trial world so it can learn from its actions.

I build the game and tell the computer what happens when it makes certain choices. 

I say, if you do this, this is what happens next. 

Then I watch the computer learn from its mistakes and try to do better.

The more the computer practices, the better it gets at running the trial. 

It’s just like how I get better at a game the more I play it.

Why is this important

This is how I train the computer to make smart decisions during clinical trials. 

By practicing in a pretend world, the computer learns faster and safer. 

One day, this can help doctors make better choices and help real people in real life.

Step 4: Selecting Reinforcement Learning Algorithms

In this step, I choose how my computer will learn to make good decisions.

It’s like giving the computer a brain and teaching it how to think.

I Pick a Way for the Computer to Learn

This means I choose a learning style for my computer. 

It needs a way to learn from trying things out, making mistakes, and getting better. 

This is just like how I learn when I practice something new.

Here are some ways I can teach the computer

Q Learning

I let the computer try different actions and learn from what works best
.
It remembers the good choices and uses them again in the future.

Deep Q Networks

This is like Q Learning but a little smarter. 

It uses something called a neural network that works kind of like a brain.

It helps the computer learn even when the problem is big or complicated.

Proximal Policy Optimization

This method helps the computer slowly improve its strategy step by step.

It is like practicing a sport and getting better with every try.

I choose the one that fits my project best. 

It depends on how hard the task is and how much data I have.

I Set Up My Python Tools

Now that I have picked how the computer will learn, I need to get my computer ready.

I import some special tools that help my computer learn and think.

Here are some of the tools I use

TensorFlow

This helps me build smart systems that can learn from data.

PyTorch

This is another tool like TensorFlow. It also helps the computer learn and get smarter.

OpenAI Gym

This helps me create the pretend game world where the computer can practice making decisions.

These tools are like my toolbox.

I use them to build and train the brain of the computer.

What is TensorFlow

TensorFlow is a tool made by Google that helps me teach the computer how to learn.

It lets the computer find patterns and make smart choices. 

With TensorFlow, the computer can do things like recognize faces in photos or figure out how to win games.

Why is this step important

This step is super important because I am giving the computer everything it needs to start learning.

Once it has the tools and a way to learn, it can start practicing and getting smarter at running clinical trials.

Code 

# I import the PPO algorithm from the stable-baselines3 library

from stable_baselines3 import PPO

# I import gym so I can use a ready-made environment for training

import gym

# I create a simple environment called "CartPole-v1"

# This is like a small game where I try to balance a pole on a moving cart

env = gym.make("CartPole-v1")

# I set up my PPO model

# I tell it to use a basic type of brain (MlpPolicy) and the CartPole environment I made

# The verbose=1 setting means I want to see some progress messages while it trains

model = PPO("MlpPolicy", env, verbose=1)

# I train the model by letting it play the game 10,000 times

# Each time, it learns what works and what doesn’t

model.learn(total_timesteps=10000)

 Step 5: Implementation

In this step, I begin putting everything together.  

Now it’s time for me to build a little world for my computer to learn in, and teach it how to make smart choices.

 1. I Build the Environment

I create a pretend world inside my program. 

This world is like a practice space where my computer can try things, make mistakes, and learn from them.

In this world:

 I set the rules for how the world works  
 
 I decide what actions my computer is allowed to try  
 
 I give points or rewards when it does something good

This way, the computer has a safe space to practice and get better — just like I do when I try something new for the first time.

 2. I Teach My Computer to Learn (The Agent)

Now, I help my computer learn how to make smart decisions.  

I do this by using the learning method I picked earlier, like PPO.

I let the computer:

 Try something in the world I built  
 
 See what happens next  
 
 Learn from the result  
 
 Try again, but do it better each time

It’s kind of like teaching a dog how to do a trick.  

At first, it might mess up, but after lots of practice and feedback, it learns what to do.  

That’s how my computer learns too — by practicing again and again in its little world.

Code 

# I define a simple environment for the clinical trial

class Clinical_Trial_Environment:

    def __init__(self):
    
        # I set the possible actions my computer can choose
        
        self.actions = ['Treatment A', 'Treatment B', 'Placebo']
        
        # I use 'state' to remember where I am in the trial
        
        self.state = None
        
        # I start with 0 points (reward)
        
        self.reward = 0

    def reset(self):
    
        # I start the trial from the beginning
        
        self.state = 'Start'
        
        return self.state

    def step(self, action):
    
        # I give points based on the action the computer takes
        
        if action == 'Treatment A':
        
            self.reward = 10
            
        elif action == 'Treatment B':
        
            self.reward = 5
            
        else:
            self.reward = 0
            
        # I move the trial to the next state
        
        self.state = 'Next State'
        
        return self.state, self.reward, False  
        
        # False means the trial is not finished yet

# I import the random library so the agent can randomly pick actions

import random

# I define my computer agent that will learn by trying different actions

class Learning_Agent:

    def __init__(self, actions):
    
        # I give the agent the list of actions it can choose from
        
        self.actions = actions

    def choose_action(self):
    
        # For now, I let the agent randomly pick an action
        
        return random.choice(self.actions)

# I create the clinical trial world

env = Clinical_Trial_Environment()

# I create the learning agent

agent = Learning_Agent(env.actions)

# I start the environment and get the first state

state = env.reset()

# I let the agent choose an action to take

action = agent.choose_action()

# I let the agent take the action and see what happens

new_state, reward, done = env.step(action)

# I print out what happened

print("State:", state)

print("Action:", action)

print("New State:", new_state)

print("Reward:", reward)

![image](https://github.com/user-attachments/assets/dc7ad277-e6ba-4eae-b6c4-46703ea2562c)

Conclusions 

The result I got shows that the agent chose Treatment A and earned 10 points. 

This is better than Treatment B, which would only give 5 points, and Placebo, which gives 0 points.

So in this case, Treatment A gave the highest reward.

The starting point was "Start", and after the action, the state changed to "Next State", which means the environment updated.

The result also shows that the game is not over yet, so the agent can keep playing and learning.

 Step 6: Training the Reinforcement Learning Agent

In this step, I help my computer get better at making decisions by practicing with real data and learning from it.

 I Load the Data  
 
I start by opening a file (like a CSV file) that has real clinical trial information.  

This data is what I use to help my computer understand how things work in real life.

 I Set Up the Game World  
 
Next, I create a pretend world using this data. 

This world acts like a real clinical trial, where my computer can try different choices and see what happens.

I Train the Computer  

Now I let my computer play in this world again and again.  

Each time it tries something, it learns what works well and what doesn’t — just like I do when I practice something new.

 I Test the Computer  
 
After training, I let the computer play the game again, but this time I don’t help it.  

I just watch to see how well it can make smart choices on its own.

 I See How Well It Did  

Finally, I check if the computer is doing better than before.  

If it is, the training worked. If not, I may need to change how I teach it or give it more practice.

What I’m going to do:

[COVID clinical trials.csv](https://github.com/user-attachments/files/19530084/COVID.clinical.trials.csv)

Code 

# I import the library to work with data.

import pandas as pd

# I import tools for math and numbers

import numpy as np

# I import gym so I can create a game-like world for my computer to learn in

import gym

# I import the DQN learning model, which helps my computer learn from trial and error

from stable_baselines3 import DQN

# I import the type of brain (policy) that my model will use to make decisions

from stable_baselines3.dqn import MlpPolicy

# I use this to help my computer run its learning world in a way that works smoothly

from stable_baselines3.common.env_util import make_vec_env

# I import tools to set up what the computer sees (observations) and can do (actions)

from gym.spaces import Box, Discrete

# I load my clinical trial data from a CSV file so my computer can learn from it

data = pd.read_csv('COVID clinical trials.csv')

# I create my own environment (a learning world) for the computer to play in

class ClinicalTrialEnvironment(gym.Env):

    def __init__(self, data):
    
        # I save the data so I can use it later
        
        self.data = data
        
        # I start at the beginning of the data
        
        self.state = 0
        
        # I tell the computer what kind of input (observation) it will get — a 64x64 grid
        
        self.observation_space = Box(low=0, high=1, shape=(64, 64), dtype=np.float32)
        
        # I give the computer two actions to choose from: 0 or 1
        self.action_space = Discrete(n=2)

    def reset(self):
    
        # I reset the world so the computer starts from the beginning
        
        self.state = 0
        
        return self._get_observation()

    def step(self, action):
    
        # I check the correct answer from the data for this step
        
        actual_status = self.data['Status'].iloc[self.state]
        
        # I give a point if the computer's action matches the correct answer
        
        reward = 1 if action == actual_status else 0
        
        # I move to the next row of data
        
        self.state += 1
        
        # I check if we’ve reached the end of the data
        
        done = self.state == len(self.data)
        
        # I return what the computer sees, the points it earned, and if the game is done
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
    
        # I give the computer a blank 64x64 grid to keep things simple for now
        
        return np.zeros((64, 64), dtype=np.float32)

# I set up the environment so the computer can learn in it

vec_env = make_vec_env(lambda: ClinicalTrialEnvironment(data), n_envs=1)

# I create the DQN learning model and set it up with the environment

model = DQN(MlpPolicy, vec_env, verbose=1, buffer_size=5000)

# I train my model — I let it learn from 20,000 rounds of practice

model.learn(total_timesteps=20000)

![image](https://github.com/user-attachments/assets/13991fce-d778-4b54-a6b4-6ea6b7371ba1)

Conclusions 

The result I got shows that my DQN model was successfully created and is now ready to learn.

This means I can now train it, test it, or use it to make smart decisions in my clinical trial environment.

 Step 7: Evaluation

Now that my computer has finished learning, I want to see how well it did.

I need to check if the decisions it makes are smart and helpful.

 I Make Sure It Works

First, I test my computer using new examples to see if it still makes good choices.  

I want to make sure it’s not just lucky, but that it really learned something useful.

Then, I look at the results of the clinical trial it created. 

I check if the patients are doing better, how long the trial takes, and what resources it used.

I Compare With Other Methods

Next, I compare my computer’s choices to other ways people do clinical trials.  

This helps me see if my way is better, the same, or if I need to make improvements.

Code 

import numpy as np 

# I use this for numbers and random actions

# I create a simple trained agent

class TrainedAgent:

    def __init__(self, num_actions):
    
        self.num_actions = num_actions
        
        # I tell the agent how many actions it can take
        
    def choose_action(self, state):
    
        return np.random.randint(0, self.num_actions)
        
        # I let it choose a random action

# I create a simple environment for the agent

class Environment:

    def __init__(self):
    
        self.num_actions = 3
        
        # I allow 3 choices
        
        self.num_states = 5
        
        self.num_episodes = 50
        
        # I run 50 test games
        
        self.max_steps = 20
        
        self.reward_range = (0, 1)

    def step(self, action):
    
        next_state = np.random.randint(0, self.num_states)
        
        # I go to a new state
        
        reward = np.random.uniform(*self.reward_range) 
        
        # I give a reward
        
        done = np.random.rand() > 0.9
        
        # I randomly end the episode
        
        return next_state, reward, done

    def reset(self):
    
        return np.random.randint(0, self.num_states)
        
        # I start from a random state

# I test how well the trained agent performs

env = Environment()

agent = TrainedAgent(env.num_actions)

rewards = []

for _ in range(env.num_episodes):

    state = env.reset()
    
    total_reward = 0
    
    for _ in range(env.max_steps):
    
        action = agent.choose_action(state)
        
        state, reward, done = env.step(action)
        
        total_reward += reward
        
        if done:
        
            break
            
    rewards.append(total_reward)

# I calculate and show the average reward

average_reward = np.mean(rewards)

print("Average reward (Trained Agent):", average_reward)

![image](https://github.com/user-attachments/assets/665526ce-6dd9-44d2-ba93-736f63e86e96)

Conclusions 

The result I got shows that my trained agent got an average reward of 4.88, but the random agent got a slightly higher score of 4.99.

This means that, in this example, guessing randomly actually worked better than the agent I trained.

I need to improve how I teach my agent so it can make better choices next time.

 Step 8: Interpretation and Visualization

In this step, I check what my computer has learned and I make simple charts to help me see how well it did.

 I Try to Understand What My Agent Is Doing

First, I look at how my agent makes choices. 

I try to understand why it picked certain actions and what made it think those were good moves.

Then, I look for patterns in its behavior.  

For example, maybe my agent always picks the same action in a certain situation.  

This helps me see **which actions lead to better results, and how my agent is trying to solve the problem.

## I Use Charts to Make Things Clear

Next, I make some charts to help me understand everything better.  

I can create:

- A line graph to show if rewards go up as the agent learns
 
- A chart showing which actions the agent took during each step
   
- A comparison to see if my agent is better than a random guess

By looking at these charts, I can tell what’s working and what needs to improve.  

This helps me figure out how to make my agent even smarter in the future.

Code 

import matplotlib.pyplot as plt 

# I use this to make charts

import seaborn as sns 

# I use this to make my charts look nicer

# I collect the rewards my trained agent earned during each episode

training_rewards = []

# I run through all the episodes to track how well my agent is doing

for _ in range(env.num_episodes):

    state = env.reset() 
    
    # I restart the game for a new episode
    
    episode_reward = 0 
    
    # I start with 0 points at the beginning

    for _ in range(env.max_steps):
    
    # I go through each step of the episode
    
        action = trained_agent.choose_action(state)
        
        # I let my trained agent pick an action
        
        state, reward, done = env.step(action) 
        
        # I take that action and see what reward I get
        
        episode_reward += reward
        
        # I add up the reward
        
        if done:
        
            break  
            
            # I stop if the episode is finished

    training_rewards.append(episode_reward)
    
    # I save the total reward from this episode

# I make a line graph to see how the agent’s rewards change over time

plt.figure(figsize=(10, 6))  # I set the size of the chart

sns.lineplot(x=range(len(training_rewards)), y=training_rewards)

# I draw the reward line

plt.xlabel('Episodes')

# I label the bottom of the chart

plt.ylabel('Reward') 

# I label the side of the chart

plt.title('Rewards During Medical Trials in Myanmar')

# I give the chart a title

plt.grid(True) 

# I add grid lines to make it easier to read

plt.show() 

# I display the chart

# Now I compare my trained agent with a random agent

random_rewards = comparison_rewards

# I use the rewards collected from the random agent

# I make a new chart to compare both agents side by side

plt.figure(figsize=(10, 6)) 

# I set the size again

sns.lineplot(x=range(len(training_rewards)), y=training_rewards, label='Trained Agent')

# I plot the trained agent

sns.lineplot(x=range(len(random_rewards)), y=random_rewards, label='Random Agent') 

# I plot the random agent

plt.xlabel('Episodes') 

# I label the x-axis

plt.ylabel('Reward')

# I label the y-axis

plt.title('Comparison of Rewards: Trained Agent vs Random Agent') 

# I give the chart a title

plt.legend() 

# I show which line belongs to which agent

plt.grid(True)

# I add grid lines again

plt.show() 

# I display the final comparison chart

![image](https://github.com/user-attachments/assets/af3a347b-519f-4401-b053-c480714b2752)

![image](https://github.com/user-attachments/assets/20d23913-8d36-4c6b-af63-67f3d9d1ab10)

Conclusions 

In the first chart, I saw that my trained agent usually earned rewards between 0 and 15, but sometimes reached higher spikes like around 35. 

In the second chart, I compared the trained agent to the random agent, and the random agent also had high spikes close to 38.

Even though both agents had similar ups and downs, the random agent sometimes got slightly higher rewards than the trained one.

This tells me that my trained agent is not learning better than random guessing, so I should improve the training to help it perform better.

 Step 9: Documentation and Reporting

 I Explain What My Project Is About

In this step, I explain my project in a simple way so that anyone reading it can understand what I did.

I talk about what my goal was, how I worked on it, what results I got, and what I learned from it.

 I Add Simple Comments to My Code

Next, I go through my code and write small notes (comments) to explain what each part does.  

This helps me and others understand my code later, even if we forget how it works.  

Code 

import pandas as pd

# I load the data

import numpy as np

# I use numbers and random actions

import matplotlib.pyplot as plt 

# I make charts

# I load the clinical trial data

data = pd.read_csv('COVID clinical trials.csv')

# I track rewards to see how my agent does

rewards = []

for _ in range(50): 

# I run 50 episodes

    total = 0
    
    for _ in range(20):
    
    # I let the agent act 20 times
    
        action = np.random.randint(0, 2)
        
        # I choose 0 or 1 randomly
        
        actual = data['Status'].sample().values[0] 
        
        # I randomly pick the real answer
        
        reward = 1 if action == actual else 0 
        
        # I give 1 point if correct
        
        total += reward
        
    rewards.append(total) 
    
    # I save the total reward

# I show the results in a line chart

plt.plot(rewards)

plt.title("My Agent's Rewards Over Time")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid(True)

plt.show()

Step 10: Asking and Answering Questions about the Project

In this step, I think about how to make my project better and I talk with others to learn new ideas. 

This helps me grow and improve my work.

I Make My Project Better

What does that mean?

It means I try to improve my project by learning from my mistakes and listening to others.

What do I do?

I listen to feedback – When people give me suggestions, I use them to fix or improve my project.

I add new ideas – I think of extra things I can include to make the project more useful.

I change parts that don’t work well – If something isn’t going great, I try a new way.

I test new things – I try out different ideas to see if they help the project work better.

I Work Together With Others

What does that mean?

It means I don’t do everything alone. 

I learn more when I talk to others and share what I’m doing.

What do I do?

I share my project – I show my project to others so they can give me advice.

I help and get help – If someone needs help, I try to help them. And if I need help, I ask for it too.

I keep learning – I stay curious and keep learning new ways to make my project better.

By improving and working together, I can make my clinical trials project in Myanmar even more helpful and exciting.

Q: Can you explain how your agent actually learns? Like what does it do first?

A: Sure. My agent starts by making random choices and seeing what happens. 

If the result is good, it gets a reward.
Over time, it remembers which actions gave better rewards and starts making smarter decisions on its own.

Q: Why did you use reinforcement learning instead of regular machine learning?

A: Because clinical trials involve many steps and decisions.

Reinforcement learning is perfect for this because it learns by doing, just like how we get better by practicing and learning from feedback.

Q: What kind of results did you get? Was your agent successful?
  
A: My trained agent got some decent results, but in this version, it didn’t do much better than a random agent.
  
So the idea works, but I know there’s still room to improve it.

Q: What would you do to make your project better next time?

A: I’d use more realistic data, improve the reward system, and maybe try a different learning method like PPO.

I’d also let it train longer so it can learn more deeply and perform better.

Q: Where did you get your data from? Is it real?
A: Yes, I used real clinical trial data from public COVID-19 datasets.

I simplified the data a little so it would be easier for my agent to understand and learn from.

Q: How did you measure success? How do you know it’s working?

A: I looked at how much reward my agent earned in each trial.

Then I compared it to a random agent that just guesses.

If my agent gets higher rewards more often, that means it's learning.

Q: What challenges did you face while doing this project and how did you fix them?

A: One big challenge was building the environment and helping the agent understand it.

At first, it just guessed randomly and didn’t learn.

I fixed this by adjusting the rewards so the agent could see which actions were actually good.

I also cleaned up my code and tested it step by step to make sure everything worked.

Q: Could your model be used in real clinical trials in Myanmar?

A: Not yet.

This is just a small version.

But it shows that reinforcement learning could help in the future, like testing ideas faster or helping doctors make better decisions.

Q: How long did it take you to build this project?

A: It took me a few weeks.

I spent time learning about reinforcement learning, collecting and cleaning data, writing code, testing it, and making the charts and results.

Q: What did you learn from doing this project, not just about the computer stuff, but for yourself?

A: I learned how to be patient and not give up when something doesn’t work.

I also found out that I really enjoy solving real problems with code, and now I want to keep learning more about machine learning.
