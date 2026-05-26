package com.example.posturedetectionapp

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.content.BroadcastReceiver
import android.content.IntentFilter
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.IOException

class PostureMonitoringService : Service(), SensorEventListener {

    private val binder = LocalBinder()

    // Service state flows
    private val _currentPosture = MutableStateFlow<PostureLabel?>(null)
    val currentPosture: StateFlow<PostureLabel?> = _currentPosture

    private val _confidence = MutableStateFlow(0f)
    val confidence: StateFlow<Float> = _confidence

    private val _latency = MutableStateFlow(0L)
    val latency: StateFlow<Long> = _latency

    private val _badPostureDuration = MutableStateFlow(0f)
    val badPostureDuration: StateFlow<Float> = _badPostureDuration

    private val _statusColor = MutableStateFlow(StatusColor.GREY)
    val statusColor: StateFlow<StatusColor> = _statusColor

    private val _isUsingQuantized = MutableStateFlow(false)
    val isUsingQuantized: StateFlow<Boolean> = _isUsingQuantized

    private val _systemStatus = MutableStateFlow("Ready")
    val systemStatus: StateFlow<String> = _systemStatus

    private val _sensorStatus = MutableStateFlow("Sensors: OK")
    val sensorStatus: StateFlow<String> = _sensorStatus

    private val _isMonitoring = MutableStateFlow(false)
    val isMonitoring: StateFlow<Boolean> = _isMonitoring

    // Core components
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private val sensorLock = Any()
    private val latestAcc = FloatArray(3)
    private val latestGyro = FloatArray(3)
    private var hasAcc = false
    private var hasGyro = false

    private val sensorRingBuffer = SensorRingBuffer(capacity = 100)
    private lateinit var normalizationConfig: NormalizationConfig
    private lateinit var classifier: TFLitePostureClassifier
    private val movingAverageSmoother = MovingAverageSmoother(windowSize = 3)

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var samplingJob: Job? = null
    private var inferenceJob: Job? = null

    private var badPostureStartTime: Long = 0
    private var lastVibrationTime: Long = 0
    private var isPausedForScreenOff = false

    private val screenStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            Log.d("PostureService", "Received broadcast: ${intent.action}")
            when (intent.action) {
                Intent.ACTION_SCREEN_OFF -> {
                    if (_isMonitoring.value && !isPausedForScreenOff) {
                        Log.d("PostureService", "Screen OFF: pausing monitoring")
                        isPausedForScreenOff = true
                        unregisterSensorsAndStopJobs()
                        _statusColor.value = StatusColor.GREY
                        _systemStatus.value = "Paused (Screen Off)"
                        updateNotification("자세 모니터링이 일시 중지되었습니다. (화면 꺼짐)", StatusColor.GREY)
                    }
                }
                Intent.ACTION_SCREEN_ON -> {
                    val km = context.getSystemService(Context.KEYGUARD_SERVICE) as android.app.KeyguardManager
                    Log.d("PostureService", "Screen ON: isKeyguardLocked = ${km.isKeyguardLocked}")
                    if (_isMonitoring.value && isPausedForScreenOff && !km.isKeyguardLocked) {
                        Log.d("PostureService", "Screen ON & unlocked: resuming monitoring")
                        isPausedForScreenOff = false
                        registerSensorsAndStartJobs()
                    }
                }
                Intent.ACTION_USER_PRESENT -> {
                    if (_isMonitoring.value && isPausedForScreenOff) {
                        Log.d("PostureService", "User Present (Unlocked): resuming monitoring")
                        isPausedForScreenOff = false
                        registerSensorsAndStartJobs()
                    }
                }
            }
        }
    }

    companion object {
        const val NOTIFICATION_ID = 1001
        const val CHANNEL_ID = "posture_alerts"
        const val EXTRA_MODEL_NAME = "extra_model_name"
    }

    inner class LocalBinder : Binder() {
        fun getService(): PostureMonitoringService = this@PostureMonitoringService
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        normalizationConfig = NormalizationConfig(this)
        classifier = TFLitePostureClassifier(this)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (accelerometer == null || gyroscope == null) {
            _sensorStatus.value = "Error: Missing IMU Sensors!"
            _systemStatus.value = "Hardware Error"
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelName = intent?.getStringExtra(EXTRA_MODEL_NAME) ?: "posture_1dcnn_int8.tflite"

        // Initialize Foreground service status and channel
        createNotificationChannel()
        val initialNotification = createNotification("자세 모니터링 준비 중...", StatusColor.GREY)
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) { // API 34+
            startForeground(NOTIFICATION_ID, initialNotification, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH)
        } else {
            startForeground(NOTIFICATION_ID, initialNotification)
        }

        if (!_isMonitoring.value) {
            startMonitoring(modelName)
        }

        return START_NOT_STICKY
    }

    private fun startMonitoring(modelName: String) {
        if (!normalizationConfig.isLoaded) {
            normalizationConfig = NormalizationConfig(this)
            if (!normalizationConfig.isLoaded) {
                updateNotification("설정 오류: normalization.json 없음", StatusColor.GREY)
                _systemStatus.value = "Config Error"
                stopSelf()
                return
            }
        }

        try {
            classifier.loadModel(modelName)
        } catch (e: IOException) {
            e.printStackTrace()
            if (modelName == "posture_1dcnn_int8.tflite") {
                _systemStatus.value = "INT8 load failed, falling back..."
                try {
                    classifier.loadModel("posture_1dcnn_float32.tflite")
                } catch (e2: IOException) {
                    updateNotification("모델 오류: posture_1dcnn_float32.tflite 없음", StatusColor.GREY)
                    _systemStatus.value = "Model Error"
                    stopSelf()
                    return
                }
            } else {
                updateNotification("모델 오류: $modelName 없음", StatusColor.GREY)
                _systemStatus.value = "Model Error"
                stopSelf()
                return
            }
        }

        _isUsingQuantized.value = classifier.getModelName() == "posture_1dcnn_int8.tflite"
        _isMonitoring.value = true

        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_USER_PRESENT)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(screenStateReceiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            registerReceiver(screenStateReceiver, filter)
        }

        registerSensorsAndStartJobs()
    }

    private fun registerSensorsAndStartJobs() {
        synchronized(sensorLock) {
            hasAcc = false
            hasGyro = false
        }
        sensorRingBuffer.clear()
        movingAverageSmoother.clear()
        badPostureStartTime = 0
        lastVibrationTime = 0
        _badPostureDuration.value = 0f
        _statusColor.value = StatusColor.ORANGE
        _currentPosture.value = null
        _confidence.value = 0f

        val registeredAcc = accelerometer?.let {
            sensorManager.registerListener(this, it, 20000)
        } ?: false
        val registeredGyro = gyroscope?.let {
            sensorManager.registerListener(this, it, 20000)
        } ?: false

        if (!registeredAcc || !registeredGyro) {
            updateNotification("오류: 센서 등록 실패", StatusColor.GREY)
            _systemStatus.value = "Sensor Error"
            stopSelf()
            return
        }

        _systemStatus.value = "Monitoring Active"
        updateNotification("자세 모니터링이 활성화되었습니다.", StatusColor.GREEN)

        // Launch Sampling Coroutine (50 Hz)
        samplingJob = serviceScope.launch(Dispatchers.Default) {
            val periodMs = 20L
            while (isActive) {
                val startTime = SystemClock.elapsedRealtime()
                var sample: FloatArray? = null
                synchronized(sensorLock) {
                    if (hasAcc && hasGyro) {
                        sample = FloatArray(6).apply {
                            System.arraycopy(latestAcc, 0, this, 0, 3)
                            System.arraycopy(latestGyro, 0, this, 3, 3)
                        }
                    }
                }
                sample?.let { raw ->
                    val normalized = normalizationConfig.normalize(raw)
                    sensorRingBuffer.addSample(normalized)
                }
                val elapsed = SystemClock.elapsedRealtime() - startTime
                val sleepTime = (periodMs - elapsed).coerceAtLeast(0L)
                delay(sleepTime)
            }
        }

        // Launch Inference Coroutine (1 Hz)
        inferenceJob = serviceScope.launch(Dispatchers.Default) {
            while (isActive) {
                delay(1000L)

                if (!sensorRingBuffer.isFull()) {
                    val count = sensorRingBuffer.getSamplesCount()
                    _systemStatus.value = "Filling Buffer ($count%)"
                    continue
                }

                val samples = sensorRingBuffer.getSamples()
                val result = classifier.classify(samples)

                if (result != null) {
                    val smoothedProbs = movingAverageSmoother.addPrediction(result.probabilities)
                    var maxIndex = 0
                    var maxVal = smoothedProbs[0]
                    for (i in 1 until 3) {
                        if (smoothedProbs[i] > maxVal) {
                            maxVal = smoothedProbs[i]
                            maxIndex = i
                        }
                    }
                    val smoothedLabel = PostureLabel.fromIndex(maxIndex)
                    val smoothedConfidence = smoothedProbs[maxIndex]

                    _currentPosture.value = smoothedLabel
                    _confidence.value = smoothedConfidence
                    _latency.value = result.latencyMs
                    _systemStatus.value = "Monitoring Active"

                    if (smoothedLabel == PostureLabel.UPRIGHT) {
                        badPostureStartTime = 0
                        _badPostureDuration.value = 0f
                        if (_statusColor.value != StatusColor.GREEN) {
                            _statusColor.value = StatusColor.GREEN
                            updateNotification("자세가 바릅니다. 좋은 자세를 유지하세요!", StatusColor.GREEN)
                        }
                    } else {
                        if (badPostureStartTime == 0L) {
                            badPostureStartTime = SystemClock.elapsedRealtime()
                        }
                        val duration = (SystemClock.elapsedRealtime() - badPostureStartTime) / 1000f
                        _badPostureDuration.value = duration

                        if (duration >= 8f) {
                            _statusColor.value = StatusColor.RED
                            triggerBackgroundAlerts(duration, smoothedLabel)
                        } else {
                            _statusColor.value = StatusColor.ORANGE
                            val remaining = (8f - duration).coerceAtLeast(0f)
                            updateNotification("자세가 바르지 않습니다. 똑바로 하세요! (${remaining.format(0)}초 후 경고)", StatusColor.ORANGE)
                        }
                    }
                } else {
                    _systemStatus.value = "Inference Failed"
                }
            }
        }
    }

    private fun unregisterSensorsAndStopJobs() {
        samplingJob?.cancel()
        samplingJob = null
        inferenceJob?.cancel()
        inferenceJob = null

        sensorManager.unregisterListener(this)

        badPostureStartTime = 0
        lastVibrationTime = 0
        _badPostureDuration.value = 0f
        _currentPosture.value = null
        _confidence.value = 0f
    }

    private fun triggerBackgroundAlerts(duration: Float, label: PostureLabel) {
        val now = SystemClock.elapsedRealtime()

        // Double pulse vibration triggered every 1 second
        if (now - lastVibrationTime >= 1000L) {
            lastVibrationTime = now
            try {
                val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                    vibratorManager.defaultVibrator
                } else {
                    @Suppress("DEPRECATION")
                    getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val effect = if (label == PostureLabel.MILD_FORWARD_FLEXION) {
                        VibrationEffect.createWaveform(longArrayOf(0, 200), -1)
                    } else {
                        VibrationEffect.createWaveform(longArrayOf(0, 400, 150, 400), -1)
                    }
                    vibrator.vibrate(effect)
                } else {
                    @Suppress("DEPRECATION")
                    if (label == PostureLabel.MILD_FORWARD_FLEXION) {
                        vibrator.vibrate(200)
                    } else {
                        vibrator.vibrate(longArrayOf(0, 400, 150, 400), -1)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // Keep updating notification with Korean text
        val durationFormatted = duration.format(0)
        updateNotification("나쁜 자세가 ${durationFormatted}초 동안 감지되었습니다. 자세를 똑바로 하세요!", StatusColor.RED)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Posture Alert"
            val descriptionText = "자세가 8초 이상 바르지 않을 경우 경고 진동 및 알림을 보냅니다."
            val importance = NotificationManager.IMPORTANCE_HIGH
            val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
                description = descriptionText
                enableVibration(false)
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(contentText: String, color: StatusColor): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val iconRes = when (color) {
            StatusColor.RED -> android.R.drawable.ic_dialog_alert
            StatusColor.ORANGE -> android.R.drawable.ic_dialog_info
            else -> android.R.drawable.ic_dialog_info
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(iconRes)
            .setContentTitle("Posture Guard")
            .setContentText(contentText)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun updateNotification(contentText: String, color: StatusColor) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notification = createNotification(contentText, color)
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun stopMonitoring() {
        if (!_isMonitoring.value) return

        try {
            unregisterReceiver(screenStateReceiver)
        } catch (e: Exception) {
            // Ignore if not registered
        }
        isPausedForScreenOff = false

        unregisterSensorsAndStopJobs()

        _isMonitoring.value = false
        _statusColor.value = StatusColor.GREY
        _systemStatus.value = "Monitoring Stopped"

        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    override fun onDestroy() {
        stopMonitoring()
        classifier.close()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onSensorChanged(event: SensorEvent) {
        synchronized(sensorLock) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                System.arraycopy(event.values, 0, latestAcc, 0, 3)
                hasAcc = true
            } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                System.arraycopy(event.values, 0, latestGyro, 0, 3)
                hasGyro = true
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // No-op
    }

    private fun Float.format(digits: Int) = String.format("%.${digits}f", this)
}
