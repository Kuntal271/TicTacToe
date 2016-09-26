"""
TicTacToe Computer game

Author: Prabhat Rayapati

Description: This is a game of Tic Tac Toe where 
a human player can play against the computer. 
The computer calculates its move based on a 
Multi Layer Perceptron trained on the dataset
tictac_multi.txt. This shallow network contains
1 hidden layer with 1000 neurons.

Enjoy!
"""

import imp

## some special effects to the heading in the terminal
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

print(color.BOLD+ color.UNDERLINE+ 
	'\n****************************\n 	TIC-TAC-TOE 	    \n****************************' + color.END)


"""
THE GAME
"""
### Welcome 

print("\nWelcome! May the Force be with you!\n")

## run the game from the GameCore.py script
import GameCore


## play again?

playagain=input(color.BOLD+'\nWould you like to start a new game? type Y/N: '+color.END)
while playagain.lower()=='y':
	print(color.BOLD+'\nNEW GAME'+color.END)
	imp.reload(GameCore)
	playagain=input('Do you want to start a new game? type Y/N: ')

if playagain.lower()=='n':
	print(color.BOLD+'\nThank you for playing.\nTo restart the game re-run the script GameTicTac.py\n'+color.END)




