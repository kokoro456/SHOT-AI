package com.shot.detection

import kotlin.math.sqrt

/**
 * 6-state Kalman filter for tennis ball tracking.
 *
 * State vector: [x, y, vx, vy, ax, ay]
 * Measurement:  [x, y]
 *
 * Handles:
 * - Temporal smoothing of detected positions
 * - Position prediction during brief detection gaps (1-2 frames)
 * - Outlier rejection (gating: ignore detections too far from prediction)
 * - Automatic reset after prolonged miss
 *
 * Runs in ~0.01ms — negligible compared to model inference.
 */
class BallKalmanFilter {

    companion object {
        private const val STATE_DIM = 6   // x, y, vx, vy, ax, ay
        private const val MEAS_DIM = 2    // x, y

        // Tuning parameters
        private const val PROCESS_NOISE_POS = 5f       // position uncertainty per frame
        private const val PROCESS_NOISE_VEL = 20f      // velocity uncertainty per frame
        private const val PROCESS_NOISE_ACC = 40f      // acceleration uncertainty per frame
        private const val MEASUREMENT_NOISE = 8f        // detection noise (pixels)
        private const val GATE_DISTANCE = 150f          // max distance to accept measurement (pixels)
        private const val MAX_PREDICT_FRAMES = 3        // predict without measurement for N frames
    }

    // State estimate [x, y, vx, vy, ax, ay]
    private val x = FloatArray(STATE_DIM)

    // Error covariance (diagonal approximation for speed)
    private val P = FloatArray(STATE_DIM)

    // Track state
    private var initialized = false
    private var missCount = 0

    /** True if the filter has a valid state (initialized and not expired) */
    val isActive: Boolean get() = initialized && missCount <= MAX_PREDICT_FRAMES

    /**
     * Predict next state (call once per frame).
     * dt = time between frames in seconds (e.g. 1/30 for 30fps).
     */
    fun predict(dt: Float = 1f / 30f) {
        if (!initialized) return

        // State transition: constant acceleration model
        // x = x + vx*dt + 0.5*ax*dt^2
        // vx = vx + ax*dt
        // ax = ax (constant)
        val dt2 = 0.5f * dt * dt
        x[0] += x[2] * dt + x[4] * dt2  // x
        x[1] += x[3] * dt + x[5] * dt2  // y
        x[2] += x[4] * dt               // vx
        x[3] += x[5] * dt               // vy

        // Process noise: increase uncertainty
        P[0] += PROCESS_NOISE_POS * dt
        P[1] += PROCESS_NOISE_POS * dt
        P[2] += PROCESS_NOISE_VEL * dt
        P[3] += PROCESS_NOISE_VEL * dt
        P[4] += PROCESS_NOISE_ACC * dt
        P[5] += PROCESS_NOISE_ACC * dt
    }

    /**
     * Update state with new measurement.
     * Returns true if measurement was accepted (within gate), false if rejected.
     */
    fun update(measX: Float, measY: Float): Boolean {
        if (!initialized) {
            // First measurement: initialize state
            x[0] = measX
            x[1] = measY
            x[2] = 0f; x[3] = 0f  // velocity unknown
            x[4] = 0f; x[5] = 0f  // acceleration unknown
            P[0] = MEASUREMENT_NOISE * 2
            P[1] = MEASUREMENT_NOISE * 2
            P[2] = PROCESS_NOISE_VEL * 4
            P[3] = PROCESS_NOISE_VEL * 4
            P[4] = PROCESS_NOISE_ACC * 4
            P[5] = PROCESS_NOISE_ACC * 4
            initialized = true
            missCount = 0
            return true
        }

        // Gating: reject outliers
        val dx = measX - x[0]
        val dy = measY - x[1]
        val dist = sqrt(dx * dx + dy * dy)
        if (dist > GATE_DISTANCE) {
            // Measurement too far from prediction — likely a different object or noise
            // If we've been missing too long, re-initialize instead
            if (missCount >= MAX_PREDICT_FRAMES) {
                reset()
                return update(measX, measY)  // Re-initialize with this measurement
            }
            return false
        }

        // Kalman gain (simplified diagonal)
        // K = P / (P + R)
        val r = MEASUREMENT_NOISE
        val kx = P[0] / (P[0] + r)
        val ky = P[1] / (P[1] + r)

        // Innovation (measurement residual)
        val innovX = measX - x[0]
        val innovY = measY - x[1]

        // Update state
        x[0] += kx * innovX
        x[1] += ky * innovY

        // Update velocity estimate from position change
        // Blend measured velocity with current estimate
        if (missCount == 0) {
            // We had a measurement last frame too — can estimate velocity
            val kv = P[2] / (P[2] + PROCESS_NOISE_VEL)
            x[2] += kv * (innovX / (1f / 30f) - x[2]) * 0.3f
            x[3] += kv * (innovY / (1f / 30f) - x[3]) * 0.3f
        }

        // Update covariance
        P[0] *= (1 - kx)
        P[1] *= (1 - ky)

        missCount = 0
        return true
    }

    /**
     * Mark frame as missed detection.
     * The filter will use prediction only.
     */
    fun markMiss() {
        missCount++
    }

    /**
     * Get current estimated position.
     * Returns null if filter is not active.
     */
    fun getPosition(): Pair<Float, Float>? {
        if (!isActive) return null
        return Pair(x[0], x[1])
    }

    /**
     * Get current estimated velocity (pixels/frame at 30fps).
     */
    fun getVelocity(): Pair<Float, Float>? {
        if (!isActive) return null
        return Pair(x[2], x[3])
    }

    /**
     * Get confidence based on uncertainty and miss count.
     * Returns 0~1 value.
     */
    fun getConfidence(): Float {
        if (!isActive) return 0f
        val uncertainty = sqrt(P[0] * P[0] + P[1] * P[1])
        val uncertaintyFactor = (1f - (uncertainty / 100f).coerceIn(0f, 0.5f))
        val missFactor = 1f - (missCount.toFloat() / (MAX_PREDICT_FRAMES + 1))
        return (uncertaintyFactor * missFactor).coerceIn(0f, 1f)
    }

    fun reset() {
        x.fill(0f)
        P.fill(0f)
        initialized = false
        missCount = 0
    }
}
