# Gym-minigrid Implementation
Training an agent using RL tabular methods, namely TD learning on gym-minigrid

gym-minigrid - https://github.com/maximecb/gym-minigrid

The agent has been trained using 3 algoriths:
  1. Sarsa0
  2. Sarsa Lambda
  3. Q-Learning

The agent was trained on 4 environments:
  1. 5x5 Empty room
  2. 6x6 Empty room
  3. 8x8 Empty room
  4. Four Rooms
  
## Resuts:

Graphs between episode and rewards and episode and steps were plotted.

### Sarsa0:

5x5 Empty room

![sarsa0_5x5](https://user-images.githubusercontent.com/88096518/136146038-0c203710-10da-49c3-8bee-04ca42677cf9.png)

6x6 Empty room

![sarsa0_6x6](https://user-images.githubusercontent.com/88096518/136146198-7f3121e5-3a1c-4f6d-ae68-cd906dac720e.png)

8x8 Empty room

![sarsa0_8x8](https://user-images.githubusercontent.com/88096518/136146527-057c1236-f004-43ad-bc70-e253004eb6c0.png)

### Sarsa-lambda

5x5 Empty room

![sarsa_lambda_5x5](https://user-images.githubusercontent.com/88096518/136146679-0ac31285-d0e5-497b-8b93-2b0010062fcb.png)

6x6 Empty room

![sarsa_lambda_6x6](https://user-images.githubusercontent.com/88096518/136146753-914d599e-da0b-49b3-8b0b-b40ec7e22354.png)

8x8 Empty room

![sarsa_lambda_8x8](https://user-images.githubusercontent.com/88096518/136146791-91f28f44-fdb7-4cb5-a5cc-3758988bcaa4.png)

Four Rooms

For alpha = 0.1

![room4_sarsa_lambda](https://user-images.githubusercontent.com/88096518/136146872-55d99781-ef6b-4492-8161-4e359dcd32b2.png)

Comparing different alpha:

![4roomdiffalpha](https://user-images.githubusercontent.com/88096518/136146926-61a95f3d-bfe8-4351-acb6-a633379f4f1b.png)

### Q-Learning:

5x5 Empty room

![5x5_qlearning](https://user-images.githubusercontent.com/88096518/136147003-195e5b59-c01f-4cbf-8ef8-4e257d2f9ee8.png)

6x6 Empty room

![6x6_qlearning](https://user-images.githubusercontent.com/88096518/136147084-2a1d9dc9-2b1b-4a64-9a37-cb098d75e2cf.png)

8x8 Empty room

![8x8_qlearning](https://user-images.githubusercontent.com/88096518/136147113-274a98c3-e0c1-45b7-a182-2c6c8a4dbb2d.png)



