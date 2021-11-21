# connect4-AI
Connect 4 lab for Computational Intelligence.
Default is player vs ai, but you can change it setting AI_VS_AI=True

## Things I tried

- MTCS (not used because didn't work very well, but the problem could be my implementation)
- Alpha-beta pruning
- Bitboard to represent the game board instead of a 2d array
- Reordering of columns to increase chance of pruning (center columns are better in general)
- Saving the states in a dictionary since different plays can end up in the same configurations
- Bonus: I tried to use John Tromp's database (https://www.openml.org/d/40668) for the openings, but I noticed that some 'draw' configurations are missing (not only the ones where the next move is forced) so for now I return 0 if a configuration is not found in the table.
