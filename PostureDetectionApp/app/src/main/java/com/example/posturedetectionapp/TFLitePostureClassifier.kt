package com.example.posturedetectionapp

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

class TFLitePostureClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var currentModelName: String? = null
    private var isModelQuantized = false

    // Quantization parameters
    private var inputScale: Float = 0f
    private var inputZeroPoint: Int = 0
    private var outputScale: Float = 0f
    private var outputZeroPoint: Int = 0

    /**
     * Load the model from assets by name.
     */
    @Throws(IOException::class)
    fun loadModel(modelName: String) {
        if (currentModelName == modelName && interpreter != null) {
            return
        }

        close()

        val modelBuffer = loadModelFile(context, modelName)
        val options = Interpreter.Options().apply {
            setNumThreads(2)
        }

        val newInterpreter = Interpreter(modelBuffer, options)
        interpreter = newInterpreter
        currentModelName = modelName

        // Inspect input/output tensor format
        val inputTensor = newInterpreter.getInputTensor(0)
        val outputTensor = newInterpreter.getOutputTensor(0)

        val inputDataType = inputTensor.dataType()
        val quantizationParams = inputTensor.quantizationParams()

        // Check if model expects byte (int8/uint8) input or has quantization params
        isModelQuantized = (inputDataType == org.tensorflow.lite.DataType.INT8 ||
                            inputDataType == org.tensorflow.lite.DataType.UINT8 ||
                            quantizationParams.scale != 0.0f)

        if (isModelQuantized) {
            inputScale = quantizationParams.scale
            inputZeroPoint = quantizationParams.zeroPoint

            val outParams = outputTensor.quantizationParams()
            outputScale = outParams.scale
            outputZeroPoint = outParams.zeroPoint
        } else {
            inputScale = 0f
            inputZeroPoint = 0
            outputScale = 0f
            outputZeroPoint = 0
        }
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Run inference on a 100x6 sample buffer.
     * @param samples 100 samples of [ax, ay, az, gx, gy, gz]
     */
    fun classify(samples: List<FloatArray>): ClassificationResult? {
        val interp = interpreter ?: return null
        if (samples.size != 100) return null

        val startTime = SystemClock.uptimeMillis()

        try {
            val probabilities = FloatArray(3)

            if (isModelQuantized) {
                // Allocate direct ByteBuffer for 1 * 100 * 6 = 600 bytes
                val inputBuffer = ByteBuffer.allocateDirect(1 * 100 * 6)
                inputBuffer.order(ByteOrder.nativeOrder())

                for (sample in samples) {
                    for (i in 0 until 6) {
                        val rawValue = sample[i]
                        val scale = if (inputScale == 0f) 1f else inputScale
                        // Quantization: q = value / scale + zero_point
                        val q = (rawValue / scale + inputZeroPoint).roundToInt()
                        // Clamp to signed byte range [-128, 127]
                        val qClamped = q.coerceIn(-128, 127).toByte()
                        inputBuffer.put(qClamped)
                    }
                }

                inputBuffer.rewind()

                // Output buffer: 1 * 3 = 3 bytes
                val outputArray = Array(1) { ByteArray(3) }
                interp.run(inputBuffer, outputArray)

                // Dequantization: prob = scale * (q - zero_point)
                val outBytes = outputArray[0]
                for (i in 0 until 3) {
                    val qOutput = outBytes[i].toInt()
                    probabilities[i] = outputScale * (qOutput - outputZeroPoint)
                }
            } else {
                // Allocate direct FloatBuffer for 1 * 100 * 6 * 4 = 2400 bytes
                val inputBuffer = ByteBuffer.allocateDirect(1 * 100 * 6 * 4)
                inputBuffer.order(ByteOrder.nativeOrder())
                val floatBuffer = inputBuffer.asFloatBuffer()

                for (sample in samples) {
                    floatBuffer.put(sample)
                }

                val outputArray = Array(1) { FloatArray(3) }
                interp.run(inputBuffer, outputArray)
                System.arraycopy(outputArray[0], 0, probabilities, 0, 3)
            }

            // Find maximum probability
            var maxIndex = 0
            var maxVal = probabilities[0]
            for (i in 1 until 3) {
                if (probabilities[i] > maxVal) {
                    maxVal = probabilities[i]
                    maxIndex = i
                }
            }

            // Standardize and bound probabilities between 0.0 and 1.0
            val normalizedProbs = FloatArray(3)
            var sum = 0f
            for (i in 0 until 3) {
                normalizedProbs[i] = probabilities[i].coerceIn(0f, 1f)
                sum += normalizedProbs[i]
            }
            if (sum > 0f) {
                for (i in 0 until 3) {
                    normalizedProbs[i] /= sum
                }
            } else {
                normalizedProbs[maxIndex] = 1.0f
            }

            val label = PostureLabel.fromIndex(maxIndex)
            val confidence = normalizedProbs[maxIndex]
            val latencyMs = SystemClock.uptimeMillis() - startTime

            return ClassificationResult(
                label = label,
                confidence = confidence,
                probabilities = normalizedProbs,
                latencyMs = latencyMs,
                isQuantized = isModelQuantized
            )
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    fun isLoaded(): Boolean = interpreter != null

    fun getModelName(): String? = currentModelName

    fun close() {
        interpreter?.close()
        interpreter = null
        currentModelName = null
    }

    data class ClassificationResult(
        val label: PostureLabel,
        val confidence: Float,
        val probabilities: FloatArray,
        val latencyMs: Long,
        val isQuantized: Boolean
    )
}
