package com.shot.detection

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * Single-frame tennis ball detector using lightweight ONNX model.
 *
 * Phase 2b: Replaces TrackNet (3-frame, 148ms) with single-frame detection (target 15-25ms).
 * Temporal tracking is handled by BallKalmanFilter, not inside the model.
 *
 * Input:  NCHW [1, 3, 192, 192] float32 (ImageNet normalized)
 * Output: [1, 1, 48, 48] float32 heatmap (sigmoid activated)
 */
class BallTrackingDetector(context: Context) {

    companion object {
        private const val MODEL_FILE = "ball_detector.onnx"
        private const val INPUT_SIZE = 192
        private const val HEATMAP_SIZE = 48  // INPUT_SIZE / 4
        private const val NUM_INPUT_PIXELS = INPUT_SIZE * INPUT_SIZE
        private const val NUM_HEATMAP_PIXELS = HEATMAP_SIZE * HEATMAP_SIZE
        private const val CONFIDENCE_THRESHOLD = 0.3f

        // ImageNet normalization
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    // Pre-allocated arrays (zero GC per frame)
    private val inputData = FloatArray(3 * NUM_INPUT_PIXELS)
    private val pixels = IntArray(NUM_INPUT_PIXELS)
    private val heatmap = FloatArray(NUM_HEATMAP_PIXELS)
    private val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

    init {
        val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        session = ortEnv.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
    }

    var lastInferenceTimeMs: Long = 0
        private set

    data class BallPosition(
        val x: Float,
        val y: Float,
        val confidence: Float,
        val detected: Boolean
    )

    /**
     * Detect ball position from a single frame.
     * No frame buffer needed — each frame is independent.
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): BallPosition {
        val startTime = System.nanoTime()

        // Resize to 192×192
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)

        // Preprocess: ARGB → NCHW float32 with ImageNet normalization
        preprocessBitmap(resized)
        if (resized !== bitmap) resized.recycle()

        // Run inference
        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), inputShape)
        val results = session.run(mapOf(inputName to inputTensor))

        // Parse heatmap
        val outputTensor = results[0] as OnnxTensor
        outputTensor.floatBuffer.get(heatmap)

        val ballPosition = parseHeatmap(imageWidth, imageHeight)

        outputTensor.close()
        inputTensor.close()
        results.close()

        lastInferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
        return ballPosition
    }

    /**
     * Convert bitmap to NCHW float array with ImageNet normalization.
     * Writes directly into pre-allocated inputData array.
     */
    private fun preprocessBitmap(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            // ARGB → normalized RGB (ImageNet)
            val r = (pixel shr 16 and 0xFF) / 255f
            val g = (pixel shr 8 and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            inputData[i] = (r - MEAN_R) / STD_R
            inputData[NUM_INPUT_PIXELS + i] = (g - MEAN_G) / STD_G
            inputData[2 * NUM_INPUT_PIXELS + i] = (b - MEAN_B) / STD_B
        }
    }

    /**
     * Parse 48×48 heatmap → ball position in original image coordinates.
     */
    private fun parseHeatmap(imageWidth: Int, imageHeight: Int): BallPosition {
        var maxVal = 0f
        var maxIdx = 0

        for (i in heatmap.indices) {
            if (heatmap[i] > maxVal) {
                maxVal = heatmap[i]
                maxIdx = i
            }
        }

        val heatmapY = maxIdx / HEATMAP_SIZE
        val heatmapX = maxIdx % HEATMAP_SIZE

        // Scale from 48×48 heatmap to original image coordinates
        // x and y are scaled independently (handles aspect ratio difference)
        val x = heatmapX.toFloat() / HEATMAP_SIZE * imageWidth
        val y = heatmapY.toFloat() / HEATMAP_SIZE * imageHeight

        return BallPosition(
            x = x, y = y,
            confidence = maxVal,
            detected = maxVal >= CONFIDENCE_THRESHOLD
        )
    }

    fun reset() {
        // No frame buffer to reset (single-frame model)
    }

    fun close() {
        session.close()
    }
}
