package com.example.posturedetectionapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class TFLiteLatencyInstrumentedTest {
    private val warmupRuns = 200
    private val measuredRuns = 10_000
    private val numThreads = 2

    @Test
    fun benchmarkFp32AndInt8Latency() {
        benchmarkModel("posture_1dcnn_float32.tflite")
        benchmarkModel("posture_1dcnn_int8.tflite")
    }

    private fun benchmarkModel(modelName: String) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
        }
        val interpreter = Interpreter(loadModelFile(modelName), options)

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        val inputType = inputTensor.dataType()
        val outputType = outputTensor.dataType()
        val inputQuant = inputTensor.quantizationParams()
        val outputQuant = outputTensor.quantizationParams()
        val sample = createDeterministicSample()
        val preparedInput = createInputBuffer(sample, inputType, inputQuant.scale, inputQuant.zeroPoint)
        val preparedOutput = createOutputBuffer(outputType)

        repeat(warmupRuns) {
            rewind(preparedInput)
            interpreter.run(preparedInput, preparedOutput)
        }

        val invokeTimesNs = LongArray(measuredRuns)
        repeat(measuredRuns) { index ->
            rewind(preparedInput)
            val start = System.nanoTime()
            interpreter.run(preparedInput, preparedOutput)
            invokeTimesNs[index] = System.nanoTime() - start
        }

        val endToEndTimesNs = LongArray(measuredRuns)
        repeat(measuredRuns) { index ->
            val start = System.nanoTime()
            val input = createInputBuffer(sample, inputType, inputQuant.scale, inputQuant.zeroPoint)
            val output = createOutputBuffer(outputType)
            interpreter.run(input, output)
            readPredictedClass(output, outputType, outputQuant.scale, outputQuant.zeroPoint)
            endToEndTimesNs[index] = System.nanoTime() - start
        }

        println(
            buildString {
                appendLine("LATENCY_BENCHMARK_START")
                appendLine("model=$modelName")
                appendLine("threads=$numThreads")
                appendLine("warmup_runs=$warmupRuns")
                appendLine("measured_runs=$measuredRuns")
                appendLine("input_type=$inputType input_quant_scale=${inputQuant.scale} input_quant_zero_point=${inputQuant.zeroPoint}")
                appendLine("output_type=$outputType output_quant_scale=${outputQuant.scale} output_quant_zero_point=${outputQuant.zeroPoint}")
                appendLine("invoke_only_ms=${summarize(invokeTimesNs)}")
                appendLine("end_to_end_ms=${summarize(endToEndTimesNs)}")
                appendLine("LATENCY_BENCHMARK_END")
            }
        )

        interpreter.close()
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val descriptor = context.assets.openFd(modelName)
        FileInputStream(descriptor.fileDescriptor).use { inputStream ->
            val channel = inputStream.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }

    private fun createDeterministicSample(): Array<FloatArray> {
        val random = Random(20260530)
        return Array(100) {
            FloatArray(6) {
                random.nextDouble(from = -2.0, until = 2.0).toFloat()
            }
        }
    }

    private fun createInputBuffer(
        sample: Array<FloatArray>,
        inputType: org.tensorflow.lite.DataType,
        scale: Float,
        zeroPoint: Int,
    ): ByteBuffer {
        return when (inputType) {
            org.tensorflow.lite.DataType.FLOAT32 -> {
                ByteBuffer.allocateDirect(100 * 6 * 4).order(ByteOrder.nativeOrder()).apply {
                    val floatBuffer: FloatBuffer = asFloatBuffer()
                    sample.forEach { floatBuffer.put(it) }
                }
            }
            org.tensorflow.lite.DataType.INT8 -> {
                ByteBuffer.allocateDirect(100 * 6).order(ByteOrder.nativeOrder()).apply {
                    val quantScale = if (scale == 0f) 1f else scale
                    sample.forEach { row ->
                        row.forEach { value ->
                            val quantized = (value / quantScale + zeroPoint).roundToInt().coerceIn(-128, 127)
                            put(quantized.toByte())
                        }
                    }
                }
            }
            else -> error("Unsupported input type: $inputType")
        }
    }

    private fun createOutputBuffer(outputType: org.tensorflow.lite.DataType): Any {
        return when (outputType) {
            org.tensorflow.lite.DataType.FLOAT32 -> Array(1) { FloatArray(3) }
            org.tensorflow.lite.DataType.INT8 -> Array(1) { ByteArray(3) }
            else -> error("Unsupported output type: $outputType")
        }
    }

    private fun rewind(input: ByteBuffer) {
        input.rewind()
        input.asFloatBuffer().rewind()
    }

    private fun readPredictedClass(
        output: Any,
        outputType: org.tensorflow.lite.DataType,
        scale: Float,
        zeroPoint: Int,
    ): Int {
        val probabilities = when (outputType) {
            org.tensorflow.lite.DataType.FLOAT32 -> (output as Array<FloatArray>)[0]
            org.tensorflow.lite.DataType.INT8 -> {
                val bytes = (output as Array<ByteArray>)[0]
                FloatArray(bytes.size) { index -> scale * (bytes[index].toInt() - zeroPoint) }
            }
            else -> error("Unsupported output type: $outputType")
        }

        var maxIndex = 0
        for (index in 1 until probabilities.size) {
            if (probabilities[index] > probabilities[maxIndex]) {
                maxIndex = index
            }
        }
        return maxIndex
    }

    private fun summarize(timesNs: LongArray): String {
        val sortedMs = timesNs.map { it / 1_000_000.0 }.sorted()
        val mean = sortedMs.average()
        val median = percentile(sortedMs, 50.0)
        val p95 = percentile(sortedMs, 95.0)
        val min = sortedMs.first()
        val max = sortedMs.last()
        return "mean=${"%.4f".format(mean)}, median=${"%.4f".format(median)}, p95=${"%.4f".format(p95)}, min=${"%.4f".format(min)}, max=${"%.4f".format(max)}"
    }

    private fun percentile(sortedValues: List<Double>, percentile: Double): Double {
        val rank = (percentile / 100.0) * (sortedValues.size - 1)
        val lower = rank.toInt()
        val upper = minOf(lower + 1, sortedValues.lastIndex)
        val fraction = rank - lower
        return sortedValues[lower] * (1.0 - fraction) + sortedValues[upper] * fraction
    }
}
