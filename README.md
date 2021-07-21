# CS295 Reinforcement Learning Mini Project
## Sample Efficient Microsatellite Attitude Control using Deep Reinforcement Learning with Unity and OpenAI Gym

## Simulation Results
The following are simulations with a trained agent using the listed methods below. Training session lasted for 17K episodes (5.1M timesteps) in about 2 full days.

### Soft-Actor Critic V1 SACv1)
![SAC](./imgs/sacv1.gif)

### Twin-Delayed DDPG (TD3)
![SAC](./imgs/td3.gif)

### Soft-Actor Critic V2 (SACv2)
![SAC](./imgs/sacv2.gif)

### TD3 with Prioritized Experience Replay (TD3-PER)
![SAC](./imgs/td3_per.gif)

## Setting up dependencies
1. Create a virtualenvironment (or use Pytorch Docker)  
   ```
    virtualenv venv -p python3.6
    source venv/bin/activate
   ```
2. Install dependencies
    `python setup.py develop`

3. Create directories needed for training  
   - **tmp:** directory of trained models
   - **unity_environments:** directory of unity executable environment
   - **wandb:** for wandb logs when training
   ```
     mkdir tmp
     mkdir unity_environments
     mkdir wandb
   ```
   
4. Download unity executable from source and extract it on folder `unity_environments`
5. Change folder permission containing the unity executable
6. Depending on the selected DRL algorithm (e.g. TD3, SAC, SACv2, TD3-PER, etc.), change the hyperparameters and environment config on the YAML file located inside `config/train`
7. Train the model
   ```
      python bin/train/train_<DRL_ALGO>.py
   ```
   
   DRL algorithms are composed of the following (included only working ones):
   - sac: Soft-Actor Critic V1
   - sacv2: Soft-Actor Critic V2
   - td3: Twin Delayed DDPG
   - td3_per: TD3 with Prioritized Experience Replay

8. Once the model is trained, change the test config on the YAML config inside `config/test`.
9. Test the simulation using the command below. It is better if the simulation is a graphical version of the previous Unity Executable
   ```
      python bin/test/test_<DRL_ALGO>.py
   ```
   
   (Optional) You can change the number of episodes inside `bin/test/test_<DRL_ALGO>.py`

## Training Results
Results of the training can be found in this wandb repository  
https://wandb.ai/jamesandrewsarmiento/microsat_17K/overview?workspace=user-jamesandrewsarmiento

## Paper
The written research paper will be available soon.
   


