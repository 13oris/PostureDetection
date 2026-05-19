package com.example.imudatacollector

import android.os.Bundle
import android.view.Surface
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.imudatacollector.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var recorder: SensorDataRecorder

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        recorder = SensorDataRecorder(
            context = this,
            updateUi = { elapsedMs, sampleCount ->
                runOnUiThread { updateRecordingUi(elapsedMs, sampleCount) }
            },
            onWarning = { msg ->
                runOnUiThread {
                    binding.tvWarning.text = msg
                    binding.tvWarning.visibility = android.view.View.VISIBLE
                    Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                }
            }
        )

        setupContextSpinner()
        setupLabelRadioGroup()
        setupButtons()
    }

    private fun setupContextSpinner() {
        val contexts = arrayOf("Sitting", "Standing", "Commuting-like", "Resting")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, contexts)
        binding.spinnerContext.adapter = adapter
        
        binding.spinnerContext.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                recorder.currentContext = contexts[position]
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
    }

    private fun setupLabelRadioGroup() {
        binding.radioGroupLabels.setOnCheckedChangeListener { _, checkedId ->
            recorder.currentLabel = when (checkedId) {
                R.id.rbUpright -> "Upright"
                R.id.rbMild -> "Mild Forward Flexion"
                R.id.rbSevere -> "Severe Looking-Down"
                else -> "Upright"
            }
        }
    }

    private fun setupButtons() {
        binding.btnCalibrate.setOnClickListener {
            if (recorder.isRecording) {
                Toast.makeText(this, "Cannot calibrate while recording", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            binding.tvCalibrationStatus.text = "Calibration: Calibrating for 30s... Please hold naturally."
            binding.tvCalibrationStatus.setTextColor(android.graphics.Color.BLUE)
            binding.btnCalibrate.isEnabled = false
            
            recorder.startCalibration {
                runOnUiThread {
                    binding.btnCalibrate.isEnabled = true
                    if (recorder.isCalibrated) {
                        binding.tvCalibrationStatus.text = "Calibration: Done (ax: ${String.format("%.2f", recorder.baselineAx)}, ay: ${String.format("%.2f", recorder.baselineAy)}, az: ${String.format("%.2f", recorder.baselineAz)})"
                        binding.tvCalibrationStatus.setTextColor(android.graphics.Color.parseColor("#388E3C"))
                    } else {
                        binding.tvCalibrationStatus.text = "Calibration: Failed"
                        binding.tvCalibrationStatus.setTextColor(android.graphics.Color.RED)
                    }
                }
            }
        }

        binding.btnStart.setOnClickListener {
            if (!recorder.isCalibrated) {
                Toast.makeText(this, "Warning: No calibration done!", Toast.LENGTH_LONG).show()
                binding.tvWarning.text = "Warning: No calibration done!"
                binding.tvWarning.visibility = android.view.View.VISIBLE
            } else {
                binding.tvWarning.visibility = android.view.View.GONE
            }

            // Update screen rotation before recording
            recorder.screenRotation = getScreenRotation()
            
            if (recorder.startRecording()) {
                binding.tvStatus.text = "Status: Recording"
                binding.tvStatus.setTextColor(android.graphics.Color.parseColor("#388E3C"))
                binding.tvSessionId.text = "Session ID: ${recorder.sessionId}"
                binding.btnStart.isEnabled = false
                binding.btnStop.isEnabled = true
                binding.btnCalibrate.isEnabled = false
            }
        }

        binding.btnStop.setOnClickListener {
            val savedFile = recorder.stopRecording()
            binding.tvStatus.text = "Status: Stopped"
            binding.tvStatus.setTextColor(android.graphics.Color.RED)
            binding.btnStart.isEnabled = true
            binding.btnStop.isEnabled = false
            binding.btnCalibrate.isEnabled = true
            
            if (savedFile != null) {
                binding.tvSaveLocation.text = "Saved to: ${savedFile.absolutePath}"
                Toast.makeText(this, "Data saved to ${savedFile.name}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateRecordingUi(elapsedMs: Long, sampleCount: Int) {
        val seconds = (elapsedMs / 1000).toInt()
        val mins = seconds / 60
        val secs = seconds % 60
        val rate = if (seconds > 0) sampleCount / seconds else 0
        
        binding.tvTimeAndRate.text = String.format("Elapsed: %02d:%02d | Rate: %d Hz (Total: %d)", mins, secs, rate, sampleCount)
        
        if (rate > 0 && rate < 40) {
            binding.tvWarning.text = "Warning: Sampling rate is low ($rate Hz)!"
            binding.tvWarning.visibility = android.view.View.VISIBLE
        }
    }

    private fun getScreenRotation(): Int {
        val display = (getSystemService(WINDOW_SERVICE) as WindowManager).defaultDisplay
        return when (display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (recorder.isRecording) {
            recorder.stopRecording()
        }
        if (recorder.isCalibrating) {
            recorder.stopCalibration()
        }
    }
}
