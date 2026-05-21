package com.example.posturedetectionapp

class MovingAverageSmoother(val windowSize: Int = 3) {
    private val history = java.util.ArrayDeque<FloatArray>()

    @Synchronized
    fun addPrediction(probabilities: FloatArray): FloatArray {
        if (history.size >= windowSize) {
            history.pollFirst()
        }
        history.addLast(probabilities.clone())

        val numClasses = probabilities.size
        val avgProbs = FloatArray(numClasses)
        for (i in 0 until numClasses) {
            var sum = 0f
            for (h in history) {
                sum += h[i]
            }
            avgProbs[i] = sum / history.size
        }
        return avgProbs
    }

    @Synchronized
    fun clear() {
        history.clear()
    }
}
