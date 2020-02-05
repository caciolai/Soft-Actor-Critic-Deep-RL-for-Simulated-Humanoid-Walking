# ReinforcementLearningProject

This is an implementation of the Reinforcement Learning algorithm SAC (as presented in [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) and [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)) for the project of the Reinforcement Learning module of the Machine Learning course of the Master in Artificial Intelligence and Robotics (A.Y. 2019/20).

The implementation is done in Python3 using the PyTorch library and has been tested for the project on the OpenAI Gym's environments [MountainCarContinuous](https://gym.openai.com/envs/MountainCarContinuous-v0/) and [Humanoid](https://gym.openai.com/envs/Humanoid-v2/).

### Prerequisites

Python3 and `pytorch` are required, as well as `gym`. I suggest an installation via Anaconda on an Ubuntu machine. 
MuJoCo is also needed for running the `gym` environments that depend on it, such as Humanoid.

0. [Optional] Create an Anaconda environment (e.g. `conda create --name gym python=3.7`, then `conda activate gym`).
1. Install MuJoCo.
    1. Install MuJoCo prerequisites.
        ```
        sudo apt-get update -y
        sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \ 
                                software-properties-common net-tools patchelf
        ```
    2. Download [MuJoCo for Linux](https://www.roboti.us/download/mujoco200_linux.zip) (license required, you can get a 30-day trial or a student license for free).
    3. Unzip the downloaded file, create a directory `~/.mujoco/`, place the unzipped folder and your license key (the `mjkey.txt` file from your email) at `~/.mujoco/`.
    4. Add to `~/.bashrc` the following lines, replacing `<username>` with your username
        ```
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco200/bin
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
        ```
    5. Install MuJoCo
        ```
        pip3 install -U 'mujoco-py<2.1,>=2.0'
        ```
2. Install Gym.
```
pip3 install gym[box2d]
```
3. Install package for rendering of environments.
```
sudo apt install ffmpeg
```
4. Install [PyTorch](https://pytorch.org/get-started/locally/).
5. Install additional Python3 packages.
```
pip3 install numpy matplotlib seaborn texttable
```

### Installing

If all the prerequisites are installed correctly, the project should be ready to run. 

Run `python3 main.py --help` to see the list of command-line options and their meaning.

### Running the tests

Run `python3 main.py --test --render --plot` to run a random agent on the selected environment. The environment should render as well as an online plot of the return per episode.

## Authors

* **Andrea Caciolai** - *Main work*
