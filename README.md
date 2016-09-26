# TicTacToe
Multilayer Perceptron powered TicTacToe

![alt tag](https://github.com/rooster06/TicTacToe/blob/master/fig1.png)

The game needs the following three files

•	**mlp_tictac.pkl**, this python object file contains the multilayer perceptron trained on the tictac_multi.txt dataset. It’s a shallow network of 1 hidden layer with 1000 neurons. This network has an accuracy of 88.44% (accuracy calculated on the dataset using 10-fold cross validation).

•	**GameCore.py**, this python script contains the main code required to for the functioning of the game.

•	**GameTicTac.py**, this is the script used to play the game. Simply run this script in the terminal and enjoy the game.

![alt tag](https://github.com/rooster06/TicTacToe/blob/master/fig2.png)

The computer plays a fairly decent game of tic tac toe with the mlp classifier for its optimal moves prediction. I couldn’t win in my ten game trails against the computer, though I dint loose either, all of the games ended in a tie. Though there were instances where the computer could have won but made a move that wasn’t the winning move and the game ended in a tie. Though this could have been because when I chose the move from the predicted optimal moves I chose the smallest of the predicted indexes that dint overlap with the occupied states on the board, for ex-
Suppose the present state of the board was 1,0,-1,1,0,0,0,0,0

And say the prediction for next optimal play was 1,1,0,0,1,1,0,0,0, this implies that the allowed next moves for the computer are at index 2,5,6. And the python script I wrote chooses the minimum of these allowed next moves i.e. index 2, which makes the state of the board after the computer move : 1,-1,-1,1,0,0,0,0,0. So maybe conditioning the script to check if ant of the allowed moves would lead to a win would fix/make the computer better. 
Though it always leads to a tie so I would say, its performing pretty well. 

**Thoughts/hypothesis:**
Though the mlp was the best of the three classifiers, I believe the KNN was only limited by the size of the dataset. If the dataset was exhaustive i.e. providing optimal game play for each and every possible intermediate state of the board then a KNN with K=1 would have the best accuracy i.e. 100 percent accuracy. Because, As the size of training data set approaches infinity in this case the exhaustive board states and next moves, the one nearest neighbour classifier guarantees an error rate of no worse than twice the Bayes error rate (the minimum achievable error rate given the distribution of the data). And since the bayes classifier is the gold standard we would achieve a 100 percent accuracy in classification, which would translate to maybe more chances of winning for the computer. This sounds logical and reasonable to me though I can’t be entirely certain it would Inc. the computers probability of winning, and the mlp already does a good job of leading to a tie all the time.

**DataSet:** Inputs are x0, ..., x8. The outputs are y0, ..., y8, where yi is 1 if the given square is an optimal move for player O, otherwise it is 0. Each line lists the xi inputs followed by the yi outputs.

The game was built in python 3.5.2 and uses scikit-learn 0.19dev 
