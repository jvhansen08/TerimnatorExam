# Notes from Walkthrough September 6 2023

Question 2:

- 80% of firetruck effort on question 2 to make the turn

  Question 3: (Should be able to get both parts)

- Calibrate the thing
- Take a video recording of a moving object and plot the probations. Describe the error and the issues there
- Will talk about in class

Question 4:

- A is a gimme point
- B solves chicken and egg problem. Tricker than question 3 algorithms to implement. Show the map and discuss it
- C Lookup the cost of all the sensors.
- D (Grad students) Orb slam algorithm outside. (Will discuss ROS in class Robot Operating System). Talk about specifically which landmarks you could see and what the distances were.
- Potentially use a QR code or marking on the ground for the robot to localize itself. (Sep 25 2023)

Question 5

- A Diagram contrasting reinforcement learning and controls
- B Cart pole problem. Teach it how to do it. Once it can work, make the right .5% more and show the video of what it looks like

Question 6:

- FER library. Gimme points (just watch video and implement)

Question 7:

- Don’t need to implement the algorithms, just use them. Compare and contrast them

Question 8:

- Classify images using whatever we want (Ex. ResNet, GoogleNet)
- Deploy YOLO on the raspberry Pi
- D shouldn’t be too much of an additional burden

## For Ros

- pynput for taking in commands
- GParted for resizing the SD card
-

## Mapping Tips (October 9 2023, 9:50 AM)
### You have a map

- Option 1) Choose to plan to the goal. If you do that, then you assume everything else is empty space and avoid seen obstacles. Assume empty space, every 1-2 seconds replan. 
- Option 2) Faster, take me to the best place given whatever horizon I can plan at. Based on either sensor range or compute stack (if we can only plan 20meters in .5 seconds, then we only have 10 seconds). Plan to the computational limit that is closest to the goal.
- Option 3) If you trust the room and the robot, then you can plan a 1 shot movement. Let the robot go, send a message back to the computer to compute where we are, and then send the location back to the robot. 
- For moving the ball: A potential field planner is not the end of the world. More dynamic in how it plans

### You don't have a map (October 9 2023, 10:00 AM)

- We don't generally trust the map we are given. Compute everything based on the sensors. All we focus on is whether there is an obstacle nearby. We are not making a map and not using as much memory.
- The more sensors on a robot the more time to process the data and setting up configuration.




# Stamp

```
                  ,
               ,  ;:._.-`''.
             ;.;'.;`      _ `.
              ',;`       ( \ ,`-.
           `:.`,         (_/ ;\  `-.
            ';:              / `.   `-._
          `;.;'              `-,/ .     `-.
          ';;'              _    `^`       `.
         ';;            ,'-' `--._          ;

': `;;        ,;     `. ':`,,.__,,_ /
 `;`:;`;:`       ,;  '.    ;,      ';';':';;`
.,; ' '-._ `':.;
            .:; ` '._ `';;,
          ;:` `    :'`' ',\_\_.)
`;:;:.,...;'`'
';. '`'::'`'' .'`'
    ,'   `';;:,..::;`'`'
, .;`      `'::''`
,`;`.

```
