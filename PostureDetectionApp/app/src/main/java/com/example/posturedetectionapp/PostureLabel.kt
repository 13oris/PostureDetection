package com.example.posturedetectionapp

enum class PostureLabel(val index: Int, val displayName: String) {
    UPRIGHT(0, "Upright"),
    MILD_FORWARD_FLEXION(1, "Mild Forward Flexion"),
    SEVERE_LOOKING_DOWN(2, "Severe Looking-Down");

    companion object {
        fun fromIndex(index: Int): PostureLabel {
            return values().firstOrNull { it.index == index } ?: UPRIGHT
        }
    }
}
