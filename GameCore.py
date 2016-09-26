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
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import time


## define a function called winner which takes a board state, and player id as inputs. human_id=1 and computer_id=-1
## and returns true or false identifying if the player won
def winner(state,p_id):
	if (state[0]==state[1]==state[2]==p_id) or (state[3]==state[4]==state[5]==p_id) \
		or (state[6]==state[7]==state[8]==p_id) or (state[0]==state[3]==state[6]==p_id) \
		or (state[1]==state[4]==state[7]==p_id) or (state[2]==state[5]==state[8]==p_id) \
		or (state[0]==state[4]==state[8]==p_id) or (state[2]==state[4]==state[6]==p_id):
		return 1
	else:
		return 0 


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

"""
Loading the multilayer Perceptron as mlp_tictac from the dill file mlp_tictac.pkl using the pickle package
"""
with open('mlp_tictac.pkl', 'rb') as f:
    mlp_tictac = pickle.load(f)




### making the board display and the state of the board

boardDisplay=np.array(['1','2','3','4','5','6','7','8','9'])

boardState=np.zeros(9)


### who starts the game

Answer = input('Would you like to make the first move? type Y/N: ')


if Answer.lower()=='y':
   print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
   ## first player move
   pMove1= int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
   if 1<=pMove1<=9:
      boardDisplay[pMove1-1]='X'
      boardState[pMove1-1]=1
      print('\nyour first move')
      print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
      ## now computers move
      print('--------------------------------\nNow my turn')
      cMove1=mlp_tictac.predict(boardState.reshape(1,-1))
      ## finding next move predictions that arent the same as players move 1
      c1=np.where(cMove1==1)
      c1=c1[0]
      c1=c1[c1!=(pMove1-1)]
      ## if prediction has atleast one diff move than the player 1 move then play the first available move
      if c1.size:
         c1=c1[0]
         boardState[c1]=-1
         boardDisplay[c1]='O'
         ## else play some random move

      else:
         c1=np.where(boardState==0)[0][0]
         boardState[c1]=-1
         boardDisplay[c1]='O'
         ## print the state of the board

      time.sleep(1)
      print('I placed a \'O\' on the board at index',c1+1)
      print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

      #pMove1 was players 1st move and c1 was computers first move
      pMove2=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
      if 1<=pMove2<=9 and (pMove2!=pMove1) and (pMove2!=c1+1):
         boardDisplay[pMove2-1]='X'
         boardState[pMove2-1]=1
         print('----------------------------------\nYour second move')
         print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
         
         ## now computers move
         print('--------------------------------\nNow my turn')
         cMove2=mlp_tictac.predict(boardState.reshape(1,-1))
         ## finding next move predictions that arent the same as players move 1,2 and computers move 1
         ## lets just remove from 2nd mlp prediction the players 1st and 2nd moves and computers 1st move by adding some gibber
         cMove2[0][pMove1-1]=100
         cMove2[0][pMove2-1]=100
         cMove2[0][c1]=100
         if sum(cMove2[0]==1):
            c2=np.where(cMove2[0]==1)[0][0]
            boardState[c2]=-1
            boardDisplay[c2]='O'

         else:
            c2=np.where(cMove2[0]==0)[0][0]
            boardDisplay[c2]='O'
            boardState[c2]=-1

         ## print the state of the board
         time.sleep(1)
         print('I placed a \'O\' on the board at index',c2+1)
         print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

         ## pMove1,pMove2 are players moves so far and c1+1 and c2+1 are computers moves so far
         pMove3=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
         if 1<=pMove3<=9 and (pMove3!=pMove2) and (pMove3!=pMove1) and (pMove3!=c1+1) and (pMove3!=c2+1):
            boardDisplay[pMove3-1]='X'
            boardState[pMove3-1]=1
            print('----------------------------------\nYour third move')
            print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

            ## since player has now made 3 moves in total, check for a win
            if winner(boardState,1):
               print(color.BOLD +color.UNDERLINE+ '\n****************\nCongratulations! You Winn!!\n****************\n' + color.END)

            else:
               ## player dint win so computer makes a move
               print('--------------------------------\nNow my turn')
               cMove3=mlp_tictac.predict(boardState.reshape(1,-1))
               ## finding next move predictions that arent the same as players move 1,2,3 and computers move 1,2
               ## lets just remove from 3rd mlp prediction the players 1st and 2nd and 3rd moves and computers 1st and 2nd moves by adding some gibber
               cMove3[0][pMove1-1]=100
               cMove3[0][pMove2-1]=100
               cMove3[0][pMove3-1]=100
               cMove3[0][c1]=100
               cMove3[0][c2]=100
               if sum(cMove3[0]==1):
                  c3=np.where(cMove3[0]==1)[0][0]
                  boardState[c3]=-1
                  boardDisplay[c3]='O'

               else:
                  c3=np.where(cMove3[0]==0)[0][0]
                  boardDisplay[c3]='O'
                  boardState[c3]=-1

               ## print the state of the board
               time.sleep(1)
               print('I placed a \'O\' on the board at index',c3+1)
               print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
               if winner(boardState,-1):
                  print(color.BOLD +color.UNDERLINE+ '\n***********************\nI Winn!!Better luck next time!\n***********************\n' + color.END)

               else:
                  ## pMove1,pMove2,pMove3 are players moves so far and c1+1,c2+1,c3+1 are computers moves so far
                  pMove4=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
                  if 1<=pMove4<=9 and (pMove4!=pMove3) and (pMove4!=pMove2) and (pMove4!=pMove1) \
                     and (pMove4!=c3+1) and (pMove4!=c1+1) and (pMove4!=c2+1):
                     boardDisplay[pMove4-1]='X'
                     boardState[pMove4-1]=1
                     print('----------------------------------\nYour fourth move')
                     print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                     ## since player has made 4 moves in total, check for win
                     if winner(boardState,1):
                        print(color.BOLD +color.UNDERLINE+ '\n****************\nCongratulations! You Winn!!\n****************\n' + color.END)

                     else:
                        ## player dint win so computers move
                        print('--------------------------------\nNow my turn')
                        cMove4=mlp_tictac.predict(boardState.reshape(1,-1))
                        ## finding next move predictions that arent the same as players move 1,2,3,4 and computers move 1,2,3
                        ## lets just remove from 4th mlp prediction the players 1st and 2nd and 3rd and 4th moves and computers 1st and 2nd and 3rd moves by adding some gibber
                        cMove4[0][pMove1-1]=100
                        cMove4[0][pMove2-1]=100
                        cMove4[0][pMove3-1]=100
                        cMove4[0][pMove4-1]=100
                        cMove4[0][c1]=100
                        cMove4[0][c2]=100
                        cMove4[0][c3]=100
                        if sum(cMove4[0]==1):
                           c4=np.where(cMove4[0]==1)[0][0]
                           boardState[c4]=-1
                           boardDisplay[c4]='O'

                        else:
                           c4=np.where(cMove4[0]==0)[0][0]
                           boardDisplay[c4]='O'
                           boardState[c4]=-1
                        ## print the state of the board
                        time.sleep(1)
                        print('I placed a \'O\' on the board at index',c4+1)
                        print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                        if winner(boardState,-1):
                           print(color.BOLD +color.UNDERLINE+ '\n***********************\nI Winn!!Better luck next time!\n***********************\n' + color.END)

                        else:
                           ## only one location left for player now, so lets fill it by defualt
                           pMove5=np.where(boardState==0)[0][0]
                           boardState[pMove5]=1
                           boardDisplay[pMove5]='X'
                           print('----------------------------------\nYour final move')
                           print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                           if winner(boardState,1):
                              print(color.BOLD +color.UNDERLINE+ '\n****************\nCongratulations! You Winn!!\n****************\n' + color.END)
                           else:
                              print(color.BOLD +color.UNDERLINE+ '\n****************\nITS A TIE!!\n****************\n' + color.END)









                  else:
                     print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
                     print('Your move is either out of bound or is already occupied')












         else: 
            print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
            print('Your move is either out of bound or is already occupied')




      else:
         ## index out of bound or your move is not accepted as index is already filled
         print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
         print('Your move is either out of bound or is already occupied')








   else: 
      ##index out of bound
      print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
      print('Your move is out of bound')


elif Answer.lower()=='n':
   ## if answer is no
   print('\nThis is a fresh board')
   print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
   ## the computer makes the first move
   print('\n-----------------------------\nI\'m making the first move!')
   cMove1=mlp_tictac.predict(boardState.reshape(1,-1))
   ## if there is a prediction then make the move else random move
   ## i chose the random move to be at the center of the board
   if sum(cMove1[0]==1):
      c1=np.where(cMove1[0]==1)[0][0]
      boardDisplay[c1]='O'
      boardState[c1]=-1

   else:
      c1=4
      boardDisplay[c1]='O'
      boardState[c1]=-1

   time.sleep(1)
   print('I placed a \'O\' on the board at index', c1+1)
   print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

   ##players first move
   pMove1= int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
   if 1<=pMove1<=9 and (pMove1-1)!=c1:
      boardDisplay[pMove1-1]='X'
      boardState[pMove1-1]=1
      print('--------------------------------\nyour first move')
      print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

      ## now computers second move, so far we have c1 computers first move
      ## and pMove1 players first move
      print('--------------------------------\nNow my turn')
      cMove2=mlp_tictac.predict(boardState.reshape(1,-1))
      ## finding the next moves predictions that arent the same as c1 and pMove1
      ## lets remove c1 and pMove1 by adding some gibber to cMove2
      cMove2[0][pMove1-1]=100
      cMove2[0][c1]=100
      if sum(cMove2[0]==1):
         c2=np.where(cMove2[0]==1)[0][0]
         boardState[c2]=-1
         boardDisplay[c2]='O'

      else:
         c2=np.where(cMove2[0]==0)[0][0]
         boardDisplay[c2]='O'
         boardState[c2]=-1

      ##print the board
      time.sleep(1)
      print('I placed a \'O\' on the board at index',c2+1)
      print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])

      ##players second move
      ##so far we have pMove1 and c1 and c2
      pMove2=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
      if 1<=pMove2<=9 and (pMove2!=pMove1) and (pMove2!=c1+1) and (pMove2!=c2+1):
         boardDisplay[pMove2-1]='X'
         boardState[pMove2-1]=1
         print('----------------------------------\nYour second move')
         print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
         ## now computers third turn
         print('--------------------------------\nNow my turn')
         cMove3=mlp_tictac.predict(boardState.reshape(1,-1))
         ## finding next move predictions that arent the same as players move 1,2 and computers move 1,2
         ## lets just remove from 3rd mlp prediction the players 1st and 2nd moves and computers 1st,2nd moves by adding some gibber
         cMove3[0][pMove1-1]=100
         cMove3[0][pMove2-1]=100
         cMove3[0][c1]=100
         cMove3[0][c2]=100
         if sum(cMove3[0]==1):
            c3=np.where(cMove3[0]==1)[0][0]
            boardState[c3]=-1
            boardDisplay[c3]='O'

         else:
            c3=np.where(cMove3[0]==0)[0][0]
            boardDisplay[c3]='O'
            boardState[c3]=-1

         ##print the board
         time.sleep(1)
         print('I placed a \'O\' on the board at index',c3+1)
         print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
         if winner(boardState,-1):
            print(color.BOLD +color.UNDERLINE+ '\n***********************\nI Winn!!Better luck next time!\n***********************\n' + color.END)

         else:
            ## so far we have pMove1,pMove2,c1,c2,c3 moves
            pMove3=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
            if 1<=pMove3<=9 and (pMove3!=pMove2) and (pMove3!=pMove1) \
               and (pMove3!=c3+1) and (pMove3!=c1+1) and (pMove3!=c2+1):
               boardDisplay[pMove3-1]='X'
               boardState[pMove3-1]=1

               print('----------------------------------\nYour third move')
               print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
               ## check if the player won
               if winner(boardState,1):
                  print(color.BOLD +color.UNDERLINE+ '\n****************\nCongratulations! You Winn!!\n****************\n' + color.END)

               else:
                  ## player dint win so continue to computers move
                  print('--------------------------------\nNow my turn')
                  cMove4=mlp_tictac.predict(boardState.reshape(1,-1))
                  ## finding next move predictions that arent the same as players move 1,2,3 and computers move 1,2,3
                  ## lets just remove from 4th mlp prediction the players 1st and 2nd and 3rd moves and computers 1st and 2nd and 3rd moves by adding some gibber
                  cMove4[0][pMove1-1]=100
                  cMove4[0][pMove2-1]=100
                  cMove4[0][pMove3-1]=100
                  cMove4[0][c1]=100
                  cMove4[0][c2]=100
                  cMove4[0][c3]=100
                  if sum(cMove4[0]==1):
                     c4=np.where(cMove4[0]==1)[0][0]
                     boardState[c4]=-1
                     boardDisplay[c4]='O'

                  else:
                     c4=np.where(cMove4[0]==0)[0][0]
                     boardDisplay[c4]='O'
                     boardState[c4]=-1
                  ## print the state of the board
                  time.sleep(1)
                  print('\nI placed a \'O\' on the board at index',c4+1)
                  print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                  if winner(boardState,-1):
                     print(color.BOLD +color.UNDERLINE+ '\n***********************\nI Winn!!Better luck next time!\n***********************\n' + color.END)

                  else:
                     ## computer dint win so continue
                     pMove4=int(input('\nenter the index of the location where you want to place an \'X\' on the board,\nindex is the numbers you see on the board:'))
                     if 1<=pMove4<=9 and (pMove4!=pMove3) and (pMove4!=pMove2) and (pMove4!=pMove1) \
                        and (pMove4!=c3+1) and (pMove4!=c1+1) and (pMove4!=c2+1) and (pMove4!=c4+1):
                        boardDisplay[pMove4-1]='X'
                        boardState[pMove4-1]=1
                        print('----------------------------------\nYour fourth move')
                        print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                        ## check for a player win
                        if winner(boardState,1):
                           print(color.BOLD +color.UNDERLINE+ '\n****************\nCongratulations! You Winn!!\n****************\n' + color.END)

                        else:
                           ## no win, last defualt move
                           c5=np.where(boardState==0)[0][0]
                           boardState[c5]=-1
                           boardDisplay[c5]='O'
                           print('----------------------------------\nMy final move')
                           print('\n',boardDisplay[0],'|',boardDisplay[1],'|',boardDisplay[2],'\n-----------\n',boardDisplay[3],'|',boardDisplay[4],'|',boardDisplay[5],'\n-----------\n',boardDisplay[6],'|',boardDisplay[7],'|',boardDisplay[8])
                           if winner(boardState,-1):
                              print(color.BOLD +color.UNDERLINE+ '\n***********************\nI Winn!!Better luck next time!\n***********************\n' + color.END)

                           else:
                              print(color.BOLD +color.UNDERLINE+ '\n****************\nITS A TIE!!\n****************\n' + color.END)

                     else:
                        print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
                        print('Your move is either out of bound or is already occupied')








            else:
               print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
               print('Your move is either out of bound or is already occupied')







      else:
         print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
         print('Your move is either out of bound or is already occupied')




   else:
      print(color.BOLD+'\nGAME ENDED coz of Rules Violation'+color.END)
      print('Your move is out of bound or is already occupied')



   
   
else:
   print(color.BOLD+'Answer not acceptable.'+color.END)
