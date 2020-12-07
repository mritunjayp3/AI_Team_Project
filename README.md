# "Meet in the Middle" Bi-Directional Heuristic Search Algorithm in Pacman Domain

This code base implements the **"Meet in the Middle"** bi-directional heuristic search algorithm described in the paper **"Bidirectional Search That Is Guaranteed to Meet in the Middle"** by  *Robert C. Holte, Ariel Felner, Guni Sharon and Nathan R. Sturtevant* in the Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16).

The code base in this repository was created towards the partial fulfillment of the final project in CSE571:Artificial Intelligence at Arizona State University Fall 2020 under Dr. Yu Zhang. 

The collaborators for the project are  *Sean Kinahan(skinahan@asu.edu), Sunjeet Jena(sjena@asu.edu), Pratyusha Kodali(pkodali1@asu.edu) and Mritunjay Pandey(mpandey4@asu.edu).*



We implement the "Meet in Middle" (MM) algorithm on the pacman search environemnt defined in **“UC Berkeley Pacman AI Projects” developed by the DeNero, J.; Klein, D.** Available: http://ai.berkeley.edu/project_overview.html.



Specifically we have developed the MM algorithm for two search problems:

1) Position Search Problems
2) Corner Search Problems


###### Position Search Problem:

In Position Search Problem the task is to find the optimal path from the start position of the pacman to the only food pallet present in the enviroment through the mazes.


*By default the code uses "manhattan heuristic" for the "MM" algorithm. To use null heuristic(zero heuristic for each node), comment out the line 197 in "meetInMiddle" function inside search.py and add the line **"return 0"** with the same indentation.*


There are six pre-defined mazes in the position search problem:

	a) tinyMaze

		To run the code on tinyMaze:
			python pacman.py -l tinyMaze -p SearchAgent -a fn=mm

	b) smallMaze
		
		To run the code on smallMaze:
			python pacman.py -l smallMaze -p SearchAgent -a fn=mm

	c) mediumMaze

		To run the code on mediumMaze:
			python pacman.py -l mediumMaze -p SearchAgent -a fn=mm

	d) bigMaze

		To run the code on bigMaze:
			python pacman.py -l bigMaze -p SearchAgent -a fn=mm

	e) contoursMaze
		
		To run the code on contoursMaze:
			python pacman.py -l contoursMaze -p SearchAgent -a fn=mm

	f) openMaze

		To run the code on openMaze:
			python pacman.py -l openMaze -p SearchAgent -a fn=mm


###### Corner Search Problem:

In the corner search problem the task is to find a path such that the pacman eats all the food pallet in each corner of the maze, i.e. four corners and four food pallets.

*By default the code uses "manhattan heuristic" for the "MM" algorithm. To use null heuristic(zero heuristic for each node), comment out the line 522 and line 548 in "meetInMiddleCornerSearch" function inside search.py and add the line **"return 0"** with the same indentation against both the commented lines.*


There are three pre-defined mazes in the corner search problem:

	a) tinyCorners:

		To run the code on tinyMaze:
			python pacman.py -l tinyCorners -p SearchAgent -a fn=mmCorner,prob=CornersProblem  

	b) mediumCorners:

		To run the code on mediumCorners:
			python pacman.py -l mediumCorners -p SearchAgent -a fn=mmCorner,prob=CornersProblem

	c) bigCorners:
		(Takes some time to find the optimal path)
		To run the code on bigCorners:
			python pacman.py -l bigCorners -p SearchAgent -a fn=mmCorner,prob=CornersProblem		
