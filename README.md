# connect4-AI
Connect 4 lab for Computational Intelligence.
Default is player vs AI, but you can change it setting AI_VS_AI=True

## Things I tried

- MTCS (not used because didn't work very well, but the problem could be my implementation)
- Alpha-beta pruning
- Bitboard to represent the game board instead of a 2d array
- Reordering of columns to increase chance of pruning (center columns are better in general)
- Avoid exploring the tree if there are forced moves
- Saving the states in a dictionary since different plays can end up in the same configurations
