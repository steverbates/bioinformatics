#R implementation of assemble_input_set.py

frame_split <- function(frame,fold) { #Approximating behavior of np.array_split in R with data frame input
	n <- nrow(frame)
	r <- n %% fold
	q <- n %/% fold
	x <- list()
	if (r != 0) {
		for (i in 1:r) {
		x[[i]] <- frame[((q+1)*(i-1)+1):((q+1)*i),] #nrow = quotient plus an extra item for a number of subframes equal to remainder
		}
	}
	for (i in (r+1):fold) {
		x[[i]] <- frame [(q*(i-1)+1):(q*i),]  # nrow = quotient for the rest of the subframes requested
	}
	return(x)
}

assemble_input_set <- function(positives,negatives,fold=1) { #to balance populations of two outcomes for twofold classificiation problems. Assumes dataframe inputs.  Fold parameter used to generate a set of samples for cross-validation
	n <- nrow(negatives)
	p <- nrow(positives)
	positives$Survival <- rep(1,nrow(positives))
	negatives$Survival <- rep(0,nrow(negatives))
	if (fold==1) {
		if (n<p) {
			x <- rbind(positives[sample(1:nrow(positives),size=nrow(negatives)),],negatives)
		}
		else {
			x <- rbind(positives,negatives[sample(1:nrow(negatives),size=nrow(positives)),])
		}
		return(x[sample(1:nrow(x)),]) #shuffling positive and negative data before output; otherwise fit function would take validation split from back end i.e. all negative data
	} else {
		if (n<p) {
			P <- frame_split(positives[sample(1:nrow(negatives)),],fold)
			N <- frame_split(negatives,fold)
		}
		else {
			P <- frame_split(positives,fold)
			N <- frame_split(negatives[sample(1:nrow(positives)),],fold)			
		}
		x <- list()
		for (i in 1:length(P)) {
			X <- rbind(P[[i]],N[[i]])
			x[[i]] <- X[sample(1:nrow(X)),]
		}
		return(x)
	}
}







