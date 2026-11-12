# Findings so far

1. **Imitation on human CFOP solves gives a working 3x3 solver.** A one-hot MLP that
   predicts the next human move (79.8% top-1), combined with a width-256 beam search,
   solves 99.9% of 1,000 held-out full scrambles in 60 face turns on average (human
   CFOP: 62), at 0.2 s per scramble on CPU.
2. **It has no depth horizon.** The team's cubetransfer GNN, trained on random walks
   from solved, stops solving past about 10 quarter turns. The CFOP-trained policy
   solves every depth from 1 to 20, because its training data starts from real
   scrambles. The trade-off: it solves short scrambles the long, CFOP way.
3. **The sticker GNN loses to the MLP on the 3x3.** 63.2% top-1 for the better GNN
   against 79.8%, 50 minutes of training against 1, and 13% of full scrambles solved
   against 81.5% at the same beam width. Giving each sticker its home face (which
   center matches its color) helped most on the cross; the GNN still underfits and is
   weakest on the last layer, where exact sticker patterns decide the move.
4. **Next-move accuracy understates a policy.** 80% top-1 is enough to solve almost
   every scramble once a beam can recover from wrong guesses, so solve rate, not
   accuracy, is the number to compare models on.
5. **The reduction pieces work but the NxN solver is not finished.** Projection to a
   3x3, move lifting and both even-N parity fixes are verified by tests (280 in
   total); center solving and edge pairing are not written yet.

**Recommendation:** use the MLP for the 3x3 stage; keep the GNN where its size-free
features matter (the NxN center and edge phases, or cubetransfer-style transfer).

**Caveat:** this folder uses its own cube convention, not SPEC.md's; see `README.md`.
Details and figures: `results/RESULTS.md`.
