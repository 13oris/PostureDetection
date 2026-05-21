package com.example.imudatacollector

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class SensorDataRecorder(
    private val context: Context,
    private val updateUi: (Long, Int) -> Unit,
    private val onWarning: (String) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    var currentLabel: String = "Upright"
    var currentContext: String = "Sitting"

    var isRecording = false
    var isCalibrating = false

    private var fileWriter: FileWriter? = null
    private var currentFile: File? = null

    var sessionId: String = ""
    private var dayId: String = ""

    private var startTimeMs: Long = 0
    private var sampleCount: Int = 0

    // Latest sensor values
    private var lastAcc: FloatArray? = null
    private var lastGyro: FloatArray? = null

    // Calibration
    var isCalibrated = false
    var baselineAx = 0f
    var baselineAy = 0f
    var baselineAz = 0f

    private var calibrationStartTimeMs: Long = 0
    private var calibrationAccSum = FloatArray(3)
    private var calibrationSampleCount = 0

    // Screen rotation
    var screenRotation: Int = 0

    init {
        if (accelerometer == null) onWarning("Accelerometer not available")
        if (gyroscope == null) onWarning("Gyroscope not available")
    }

    fun startCalibration(onFinished: () -> Unit) {
        if (accelerometer == null) {
            onWarning("Accelerometer unavailable for calibration.")
            return
        }
        isCalibrating = true
        calibrationStartTimeMs = SystemClock.elapsedRealtime()
        calibrationAccSum = FloatArray(3)
        calibrationSampleCount = 0
        sensorManager.registerListener(this, accelerometer, 20_000)

        Thread {
            Thread.sleep(30_000)
            if (isCalibrating) {
                isCalibrating = false
                sensorManager.unregisterListener(this)
                if (calibrationSampleCount > 0) {
                    baselineAx = calibrationAccSum[0] / calibrationSampleCount
                    baselineAy = calibrationAccSum[1] / calibrationSampleCount
                    baselineAz = calibrationAccSum[2] / calibrationSampleCount
                    isCalibrated = true
                }
                // Call back on main thread (must be handled by caller if UI update is needed)
                onFinished()
            }
        }.start()
    }

    fun stopCalibration() {
        isCalibrating = false
        sensorManager.unregisterListener(this)
    }

    fun startRecording(): Boolean {
        if (accelerometer == null && gyroscope == null) {
            onWarning("Both sensors unavailable. Cannot record.")
            return false
        }

        sessionId = UUID.randomUUID().toString().substring(0, 8)
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val dayFormat = SimpleDateFormat("yyyyMMdd", Locale.US)
        val now = Date()
        val timeString = dateFormat.format(now)
        dayId = dayFormat.format(now)

        val filename = "posture_session_${sessionId}_${timeString}.csv"
        val dir = context.getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS)
        if (dir != null && !dir.exists()) {
            dir.mkdirs()
        }
        currentFile = File(dir, filename)

        try {
            fileWriter = FileWriter(currentFile)
            // CSV header
            fileWriter?.append("timestamp_ns,system_time_ms,session_id,day_id,label,context,calibrated,baseline_ax,baseline_ay,baseline_az,ax,ay,az,gx,gy,gz,screen_rotation,note\n")
        } catch (e: Exception) {
            e.printStackTrace()
            onWarning("Failed to create file: ${e.message}")
            return false
        }

        isRecording = true
        startTimeMs = SystemClock.elapsedRealtime()
        sampleCount = 0
        lastAcc = null
        lastGyro = null

        accelerometer?.let { sensorManager.registerListener(this, it, 20_000) }
        gyroscope?.let { sensorManager.registerListener(this, it, 20_000) }

        return true
    }

    fun stopRecording(): File? {
        isRecording = false
        sensorManager.unregisterListener(this)
        try {
            fileWriter?.flush()
            fileWriter?.close()
            fileWriter = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return currentFile
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        if (isCalibrating) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                calibrationAccSum[0] += event.values[0]
                calibrationAccSum[1] += event.values[1]
                calibrationAccSum[2] += event.values[2]
                calibrationSampleCount++
            }
            return
        }

        if (!isRecording) return

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            lastAcc = event.values.clone()
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            lastGyro = event.values.clone()
        }

        // To synchronize rows, write upon receiving Accelerometer, using the latest Gyroscope.
        // If accelerometer is missing, we write on Gyroscope events.
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            if (gyroscope == null || lastGyro != null) {
                writeRow(event.timestamp)
            }
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE && accelerometer == null) {
            writeRow(event.timestamp)
        }
    }

    private fun writeRow(timestampNs: Long) {
        val ax = lastAcc?.get(0) ?: 0f
        val ay = lastAcc?.get(1) ?: 0f
        val az = lastAcc?.get(2) ?: 0f

        val gx = lastGyro?.get(0) ?: 0f
        val gy = lastGyro?.get(1) ?: 0f
        val gz = lastGyro?.get(2) ?: 0f

        val sysTime = System.currentTimeMillis()
        val calibInt = if (isCalibrated) 1 else 0

        val row = "$timestampNs,$sysTime,$sessionId,$dayId,$currentLabel,$currentContext,$calibInt,$baselineAx,$baselineAy,$baselineAz,$ax,$ay,$az,$gx,$gy,$gz,$screenRotation,\n"

        try {
            fileWriter?.append(row)
            sampleCount++

            // Update UI periodically (~1 sec if 50Hz)
            if (sampleCount % 50 == 0) {
                val elapsed = SystemClock.elapsedRealtime() - startTimeMs
                updateUi(elapsed, sampleCount)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
