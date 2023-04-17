# Safe Finger
python train.py --task=ShadowHandCatchOver2Underarm_Safe_finger --algo=ppol --headless --cost_lim 25.0 --max_iterations=1000000 --rl_device=cuda:0
python train.py --task=ShadowHandOver_Safe_finger --algo=ppol --headless --cost_lim 25.0 --max_iterations=1000000 --rl_device=cuda:0  
# Safe Joint
python train.py --task=ShadowHandCatchOver2Underarm_Safe_joint --algo=ppol --headless --cost_lim 25.0 --max_iterations=1000000 --rl_device=cuda:0
python train.py --task=ShadowHandOver_Safe_joint --algo=ppol --headless --cost_lim 25.0 --max_iterations=1000000 --rl_device=cuda:0

# Die Rotation
python train.py --task=ShadowHandDieRotation --algo=ppol --cost_lim 1.0 --max_iterations=1000000 --rl_device=cuda:0 --headless  --num_envs=2048  

python train.py --task=ShadowHandDieRotation --algo=p3o --cost_lim 1.0 --max_iterations=1000000 --rl_device=cuda:1 --headless  --num_envs=2048  

python train.py --task=ShadowHandDieRotation --algo=focops --cost_lim 1.0 --max_iterations=1000000 --rl_device=cuda:2 --headless  --num_envs=2048  

python train.py --task=ShadowHandDieRotation --algo=cppo_pid --cost_lim 1.0 --max_iterations=1000000 --rl_device=cuda:3  --headless  --num_envs=2048  

# Grasp
python train.py --task=ShadowHandGrasp --algo=ppol --cost_lim 25.0 --max_iterations=1000000 --rl_device=cuda:0 --debug --num_envs=1


# Hand Over Wall & Wall Down
python train.py --task=ShadowHandOverWall --algo=ppol --cost_lim 3 --max_iterations=1000000 --rl_device=cuda:0 --headless
python train.py --task=ShadowHandOverWallDown --algo=ppol --headless --cost_lim 3 --max_iterations=1000000 --rl_device=cuda:0

# Hand Over Wall House & Wall Down House
python train.py --task=ShadowHandOverWallHouse --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0
python train.py --task=ShadowHandOverWallDownHouse --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0 --num_envs=1
# Hand Over Wall PC
python train.py --task=ShadowHandOverWallPC --algo=ppol --headless --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0
# Hand Over Wall PC NEW
python train.py --task=ShadowHandOverWallPCNew --algo=ppol --headless --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:7 --headless
# Catch Under Arm
python train.py --task=ShadowHandCatchUnderarmWall --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0 --num_envs=2 
python train.py --task=ShadowHandCatchUnderarmWallDown --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0 --num_envs=2 
# Pick Bottle
python train.py --task=ShadowHandPickBottle --algo=ppol --headless --cost_lim 13 --max_iterations=1000000 --rl_device=cuda:0
# Pick Bottle House
python train.py --task=ShadowHandPickBottleHouse --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0
# Jenga
python train.py --task=ShadowHandJenga --algo=focops --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:5 --headless
# Jenga(shadow hand and allegro hand)
python train.py --task=ShadowHandAllegroHandJenga --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0 --num_envs=4 
# JengaHouse(shadow hand and allegro hand)
python train.py --task=ShadowHandAllegroHandJengaHouse --algo=ppol --cost_lim 0.5 --max_iterations=10000 --rl_device=cuda:0 --num_envs=2

# JengaHousePC(shadow hand and allegro hand)
python train.py --task=ShadowHandAllegroHandJengaHousePC --algo=ppol --cost_lim 0.5 --max_iterations=10000 --rl_device=cuda:0 --num_envs=2 --headless

# Jenga (shadow hand+ allegro with arm)
python train.py --task=ShadowHandAllegroHandArmJenga --algo=ppol --cost_lim 0.5 --max_iterations=1000000 --rl_device=cuda:0 --num_envs=4

# Clean
python train.py --task=ShadowHandClean --algo=ppol --headless --cost_lim 100 --max_iterations=1000000 --rl_device=cuda:0
