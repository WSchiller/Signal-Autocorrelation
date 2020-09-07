# Signal-Autocorrelation
Autocorrelation is the correlation of a signal with a delayed copy of itself as a function of delay.  Simply, it is the similarity between observations as a function of time lag between them.  This project takes values from a noisy signal, i.e., an array of length Size random-looking values sampled over time.  Each array element is multiplied by itself and them added up.  

Sums[0] = A[0]*A[0] + A[1]*A[1] + ... + A[Size-1]*A[Size-1]

The array is shifted over one index and pairwise multiplication is performed again.

Sums[1] = A[0]*A[1] + A[1]*A[2] + ... + A[Size-1]*A[0]

The process is repeated until all pairwise multiplication shifts are completed.  The resulting sums are graphed as a function.  If there is a secret harmonic frequency hidden in the signal, there will be maximum where the period of the harmonic signal is.  There will be a minimum where the half period is.  Autocorrelation is used by scientists and engineers to see if there are regular patterns in a signal.  This problem is ideal for parallel computing as these signals can be quite large.  SIMD, OpenMP, and CUDA were used to perform the calculations and compare performances.   
 
