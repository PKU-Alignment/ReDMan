
Files and actions for EZGripper modeling

Launch Files
- display_gen2.launch  <-launches an EZGripper Dual Gen2 in Rviz->
- display.launch  <-launches an EZGripper Dual Gen1 in Rviz->

Meshes Files
- ezgripper_gen2 directory of files.  These are used for collision and visual.

URDF Files
- ezgripper_dual_gen2_articulated.urdf.xacro

How to launch Rviz

$ roslaunch ezgripper_driver display_gen2.launch    <-launches gen2 quad configuration->

or 

$ roslaunch ezgripper_driver display.launch   <-launches gen1 quad configuration->

