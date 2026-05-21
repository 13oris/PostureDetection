package com.example.posturedetectionapp

import android.content.Context
import org.json.JSONObject
import java.io.IOException

class NormalizationConfig(context: Context) {
    val mean = FloatArray(6)
    val std = FloatArray(6)
    var isLoaded = false
        private set

    init {
        try {
            val jsonString = context.assets.open("normalization.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            val meanArray = jsonObject.getJSONArray("mean")
            val stdArray = jsonObject.getJSONArray("std")

            for (i in 0 until 6) {
                mean[i] = meanArray.getDouble(i).toFloat()
                std[i] = stdArray.getDouble(i).toFloat()
            }
            isLoaded = true
        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun normalize(rawSample: FloatArray): FloatArray {
        if (!isLoaded) return rawSample
        val normalized = FloatArray(6)
        for (i in 0 until 6) {
            // Avoid division by zero if std is somehow 0
            val stdValue = if (std[i] == 0f) 1f else std[i]
            normalized[i] = (rawSample[i] - mean[i]) / stdValue
        }
        return normalized
    }
}
