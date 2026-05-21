package com.example.posturedetectionapp

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.*
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import com.example.posturedetectionapp.ui.theme.PostureDetectionAppTheme
import kotlinx.coroutines.*
import kotlin.math.roundToInt

enum class StatusColor {
    GREEN,  // Upright
    ORANGE, // Bad posture warning
    RED,    // Bad posture alert (>= 8s)
    GREY    // Stopped / Idle
}

class MainActivity : ComponentActivity() {

    private var postureService: PostureMonitoringService? = null
    private var isBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as PostureMonitoringService.LocalBinder
            postureService = binder.getService()
            isBound = true
            observeServiceStates()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            postureService = null
            isBound = false
        }
    }

    private var serviceObservationJob: Job? = null

    // App core objects
    private lateinit var normalizationConfig: NormalizationConfig

    // Compose states exposed to UI
    private val isMonitoringState = mutableStateOf(false)
    private val currentPostureState = mutableStateOf<PostureLabel?>(null)
    private val confidenceState = mutableStateOf(0f)
    private val latencyState = mutableStateOf(0L)
    private val badPostureDurationState = mutableStateOf(0f)
    private val statusColorState = mutableStateOf(StatusColor.GREY)
    private val selectedModelIndexState = mutableStateOf(0) // 0 = INT8, 1 = FLOAT32
    private val isUsingQuantizedState = mutableStateOf(false)
    private val systemStatusState = mutableStateOf("Ready")
    private val sensorStatusState = mutableStateOf("Sensors: OK")
    private val showDialogState = mutableStateOf(false)
    private val dialogMessageState = mutableStateOf("")

    // Model Filenames
    private val modelInt8 = "posture_1dcnn_int8.tflite"
    private val modelFloat32 = "posture_1dcnn_float32.tflite"

    // Launcher for notification permission (Android 13+)
    private val requestNotificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            Toast.makeText(this, "Notification permission granted", Toast.LENGTH_SHORT).show()
            startMonitoring()
        } else {
            Toast.makeText(this, "Notification permission denied. Push alarms disabled.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Init config
        normalizationConfig = NormalizationConfig(this)

        setContent {
            PostureDetectionAppTheme {
                PostureDashboardScreen()
            }
        }
    }

    override fun onStart() {
        super.onStart()
        // Bind to service to reconnect if it's already running in background
        val intent = Intent(this, PostureMonitoringService::class.java)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
            postureService = null
            serviceObservationJob?.cancel()
            serviceObservationJob = null
        }
    }

    private fun observeServiceStates() {
        val service = postureService ?: return
        serviceObservationJob?.cancel()
        serviceObservationJob = lifecycleScope.launch {
            launch {
                service.isMonitoring.collect { isMonitoringState.value = it }
            }
            launch {
                service.currentPosture.collect { currentPostureState.value = it }
            }
            launch {
                service.confidence.collect { confidenceState.value = it }
            }
            launch {
                service.latency.collect { latencyState.value = it }
            }
            launch {
                service.badPostureDuration.collect { badPostureDurationState.value = it }
            }
            launch {
                service.statusColor.collect { statusColorState.value = it }
            }
            launch {
                service.isUsingQuantized.collect { isUsingQuantizedState.value = it }
            }
            launch {
                service.systemStatus.collect { systemStatusState.value = it }
            }
            launch {
                service.sensorStatus.collect { sensorStatusState.value = it }
            }
        }
    }

    private fun startMonitoring() {
        // Check if config loaded
        if (!normalizationConfig.isLoaded) {
            normalizationConfig = NormalizationConfig(this)
            if (!normalizationConfig.isLoaded) {
                showDialogMessage("Missing configuration file: Place normalization.json in assets directory before starting.")
                return
            }
        }

        // Determine which model to load
        val activeModelName = if (selectedModelIndexState.value == 0) modelInt8 else modelFloat32
        try {
            assets.openFd(activeModelName).close()
        } catch (e: Exception) {
            if (selectedModelIndexState.value == 0) {
                // If INT8 fails, attempt fallback to FLOAT32 automatically
                try {
                    assets.openFd(modelFloat32).close()
                    selectedModelIndexState.value = 1
                    Toast.makeText(this, "INT8 model missing. Switched to Float32 debug model.", Toast.LENGTH_LONG).show()
                } catch (e2: Exception) {
                    showDialogMessage("Missing model files: Please copy posture_1dcnn_int8.tflite or posture_1dcnn_float32.tflite into assets.")
                    return
                }
            } else {
                showDialogMessage("Missing model file: $activeModelName not found in assets.")
                return
            }
        }

        // Request notification permission dynamically for Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                requestNotificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                return
            }
        }

        // Start Foreground Service
        val serviceIntent = Intent(this, PostureMonitoringService::class.java).apply {
            putExtra(PostureMonitoringService.EXTRA_MODEL_NAME, if (selectedModelIndexState.value == 0) modelInt8 else modelFloat32)
        }
        ContextCompat.startForegroundService(this, serviceIntent)

        // Bind to it immediately
        if (!isBound) {
            bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
        } else {
            observeServiceStates()
        }
    }

    private fun stopMonitoring() {
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
            postureService = null
            serviceObservationJob?.cancel()
            serviceObservationJob = null
        }

        val serviceIntent = Intent(this, PostureMonitoringService::class.java)
        stopService(serviceIntent)

        // Reset UI states
        isMonitoringState.value = false
        statusColorState.value = StatusColor.GREY
        currentPostureState.value = null
        confidenceState.value = 0f
        badPostureDurationState.value = 0f
        systemStatusState.value = "Monitoring Stopped"
    }

    private fun showDialogMessage(message: String) {
        dialogMessageState.value = message
        showDialogState.value = true
    }

    // Jetpack Compose User Interface
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun PostureDashboardScreen() {
        val isMonitoring by isMonitoringState
        val currentPosture by currentPostureState
        val confidence by confidenceState
        val latency by latencyState
        val badPostureDuration by badPostureDurationState
        val statusColor by statusColorState
        val selectedModelIndex by selectedModelIndexState
        val isUsingQuantized by isUsingQuantizedState
        val systemStatus by systemStatusState
        val sensorStatus by sensorStatusState
        val showDialog by showDialogState
        val dialogMessage by dialogMessageState

        // Animation configurations for neon status indicators
        val uiColor = when (statusColor) {
            StatusColor.GREEN -> Color(0xFF0D9488)  // Emerald teal
            StatusColor.ORANGE -> Color(0xFFD97706) // Sunset Amber
            StatusColor.RED -> Color(0xFFE11D48)    // Neon Red/Rose
            StatusColor.GREY -> Color(0xFF64748B)   // Slate
        }

        val animatedColor by animateColorAsState(
            targetValue = uiColor,
            animationSpec = tween(durationMillis = 500, easing = LinearOutSlowInEasing),
            label = "UI Color Animation"
        )

        // Pulsing animation for red alert state
        val infiniteTransition = rememberInfiniteTransition(label = "Pulsing Transition")
        val pulseScale by infiniteTransition.animateFloat(
            initialValue = 1.0f,
            targetValue = if (statusColor == StatusColor.RED) 1.06f else 1.0f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 700, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "Pulsing Scale Animation"
        )

        // Scaffold with Dark Theme Background
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            topBar = {
                CenterAlignedTopAppBar(
                    title = {
                        Text(
                            text = "POSTURE GUARD",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp,
                            letterSpacing = 2.sp
                        )
                    },
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        containerColor = Color(0xFF0F172A)
                    )
                )
            }
        ) { innerPadding ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.verticalGradient(
                            listOf(Color(0xFF0F172A), Color(0xFF1E1B4B))
                        )
                    )
                    .padding(innerPadding)
                    .padding(horizontal = 24.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(bottom = 24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    // 1. Model Selection Tabs at the top
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        TabRow(
                            selectedTabIndex = selectedModelIndex,
                            containerColor = Color.White.copy(alpha = 0.05f),
                            contentColor = Color.White,
                            indicator = { tabPositions ->
                                TabRowDefaults.SecondaryIndicator(
                                    modifier = Modifier.tabIndicatorOffset(tabPositions[selectedModelIndex]),
                                    color = animatedColor
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                        ) {
                            Tab(
                                selected = selectedModelIndex == 0,
                                onClick = {
                                    if (!isMonitoring) {
                                        selectedModelIndexState.value = 0
                                    }
                                },
                                text = {
                                    Text(
                                        "INT8 (Primary)",
                                        fontWeight = FontWeight.SemiBold,
                                        color = if (selectedModelIndex == 0) Color.White else Color.Gray
                                    )
                                },
                                enabled = !isMonitoring
                            )
                            Tab(
                                selected = selectedModelIndex == 1,
                                onClick = {
                                    if (!isMonitoring) {
                                        selectedModelIndexState.value = 1
                                    }
                                },
                                text = {
                                    Text(
                                        "Float32 (Fallback)",
                                        fontWeight = FontWeight.SemiBold,
                                        color = if (selectedModelIndex == 1) Color.White else Color.Gray
                                    )
                                },
                                enabled = !isMonitoring
                            )
                        }
                        if (isMonitoring) {
                            Text(
                                text = "Model locked during monitoring",
                                color = Color.Gray,
                                fontSize = 11.sp,
                                modifier = Modifier.padding(top = 4.dp)
                            )
                        }
                    }

                    // 2. Center Ring Gauge
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .size(260.dp)
                            .scale(pulseScale)
                            .clip(CircleShape)
                            .background(Color.White.copy(alpha = 0.02f))
                            .border(
                                width = 8.dp,
                                brush = Brush.radialGradient(
                                    colors = listOf(
                                        animatedColor,
                                        animatedColor.copy(alpha = 0.4f)
                                    )
                                ),
                                shape = CircleShape
                            )
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center,
                            modifier = Modifier.padding(24.dp)
                        ) {
                            if (isMonitoring) {
                                if (currentPosture != null) {
                                    Text(
                                        text = currentPosture!!.displayName,
                                        color = Color.White,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 24.sp,
                                        textAlign = TextAlign.Center,
                                        lineHeight = 28.sp
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "Confidence: ${(confidence * 100).roundToInt()}%",
                                        color = Color.White.copy(alpha = 0.7f),
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                } else {
                                    Text(
                                        text = "BUFFERING",
                                        color = animatedColor,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 20.sp,
                                        letterSpacing = 1.sp
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Accumulating data...",
                                        color = Color.Gray,
                                        fontSize = 12.sp,
                                        textAlign = TextAlign.Center
                                    )
                                }
                            } else {
                                Text(
                                    text = "OFFLINE",
                                    color = Color.Gray,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 26.sp,
                                    letterSpacing = 2.sp
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Monitoring inactive",
                                    color = Color.Gray.copy(alpha = 0.6f),
                                    fontSize = 12.sp
                                )
                            }
                        }
                    }

                    // 3. Warning Countdown (Sustained Poor Posture Timer)
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        if (isMonitoring && (statusColor == StatusColor.ORANGE || statusColor == StatusColor.RED)) {
                            Text(
                                text = if (statusColor == StatusColor.RED)
                                    "POOR POSTURE DETECTED!"
                                else
                                    "Slouch Alert in: ${((8f - badPostureDuration).coerceAtLeast(0f)).format(1)}s",
                                color = animatedColor,
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            // Countdown Linear Progress Indicator
                            LinearProgressIndicator(
                                progress = { (badPostureDuration / 8f).coerceIn(0f, 1f) },
                                modifier = Modifier
                                    .fillMaxWidth(0.85f)
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                                color = animatedColor,
                                trackColor = Color.White.copy(alpha = 0.1f),
                            )
                        } else if (isMonitoring && statusColor == StatusColor.GREEN) {
                            Text(
                                text = "Posture is excellent!",
                                color = animatedColor,
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp
                            )
                        } else {
                            Spacer(modifier = Modifier.height(28.dp))
                        }
                    }

                    // 4. Metrics Dashboard Grid
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            // Card 1: System Status
                            MetricCard(
                                title = "Status",
                                value = systemStatus,
                                subValue = sensorStatus,
                                modifier = Modifier.weight(1f)
                            )
                            // Card 2: Latency
                            MetricCard(
                                title = "Latency",
                                value = if (isMonitoring && latency > 0) "${latency} ms" else "--",
                                subValue = if (isMonitoring) (if (isUsingQuantized) "INT8 Quantized" else "Float32 Mode") else "Inactive",
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }

                    // 5. Control Action Buttons at the bottom
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 16.dp),
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Button(
                            onClick = { startMonitoring() },
                            enabled = !isMonitoring,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF0F766E), // Dark emerald teal
                                disabledContainerColor = Color(0xFF0F766E).copy(alpha = 0.2f),
                                contentColor = Color.White,
                                disabledContentColor = Color.White.copy(alpha = 0.4f)
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .height(56.dp)
                                .border(
                                    width = 1.dp,
                                    color = if (!isMonitoring) Color(0xFF0D9488) else Color.Transparent,
                                    shape = RoundedCornerShape(28.dp)
                                ),
                            shape = RoundedCornerShape(28.dp)
                        ) {
                            Text(
                                text = "START",
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.5.sp,
                                fontSize = 16.sp
                            )
                        }

                        Button(
                            onClick = { stopMonitoring() },
                            enabled = isMonitoring,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF9F1239), // Rich rose red
                                disabledContainerColor = Color(0xFF9F1239).copy(alpha = 0.2f),
                                contentColor = Color.White,
                                disabledContentColor = Color.White.copy(alpha = 0.4f)
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .height(56.dp)
                                .border(
                                    width = 1.dp,
                                    color = if (isMonitoring) Color(0xFFE11D48) else Color.Transparent,
                                    shape = RoundedCornerShape(28.dp)
                                ),
                            shape = RoundedCornerShape(28.dp)
                        ) {
                            Text(
                                text = "STOP",
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.5.sp,
                                fontSize = 16.sp
                            )
                        }
                    }
                }
            }
        }

        // Dialog for alerts/warnings (e.g. missing files)
        if (showDialog) {
            AlertDialog(
                onDismissRequest = { showDialogState.value = false },
                title = {
                    Text(
                        text = "Setup Required",
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                },
                text = {
                    Text(
                        text = dialogMessage,
                        color = Color.LightGray,
                        fontSize = 14.sp
                    )
                },
                confirmButton = {
                    TextButton(onClick = { showDialogState.value = false }) {
                        Text(text = "OK", color = animatedColor, fontWeight = FontWeight.Bold)
                    }
                },
                containerColor = Color(0xFF1E293B), // Dark slate
                shape = RoundedCornerShape(16.dp)
            )
        }
    }

    @Composable
    fun MetricCard(
        title: String,
        value: String,
        subValue: String,
        modifier: Modifier = Modifier
    ) {
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(16.dp))
                .background(Color.White.copy(alpha = 0.03f))
                .border(
                    width = 1.dp,
                    color = Color.White.copy(alpha = 0.08f),
                    shape = RoundedCornerShape(16.dp)
                )
                .padding(16.dp)
        ) {
            Text(
                text = title.uppercase(),
                color = Color.Gray,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = value,
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = subValue,
                color = Color.Gray,
                fontSize = 11.sp,
                lineHeight = 14.sp
            )
        }
    }

    // Quick extension to format durations
    private fun Float.format(digits: Int) = String.format("%.${digits}f", this)
}