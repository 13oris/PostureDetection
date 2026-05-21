package com.example.posturedetectionapp

class SensorRingBuffer(val capacity: Int = 100) {
    private val buffer = java.util.ArrayDeque<FloatArray>(capacity)

    @Synchronized
    fun addSample(sample: FloatArray) {
        if (buffer.size >= capacity) {
            buffer.pollFirst()
        }
        // Clone the array to avoid reference leaking and race conditions
        buffer.addLast(sample.clone())
    }

    @Synchronized
    fun isFull(): Boolean = buffer.size == capacity

    @Synchronized
    fun getSamples(): List<FloatArray> = buffer.toList()

    @Synchronized
    fun getSamplesCount(): Int = buffer.size

    @Synchronized
    fun clear() {
        buffer.clear()
    }
}
