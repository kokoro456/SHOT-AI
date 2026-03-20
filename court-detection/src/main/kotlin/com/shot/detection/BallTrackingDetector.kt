package com.shot.detection

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * Detects tennis ball position using TrackNet ONNX model.
 *
 * Takes 3 consecutive frames as input and outputs a heatmap
 * indicating the ball's probable location.
 *
 * Input:  NCHW [1, 9, 128, 320] float32 (3 RGB frames concatenated)
 * Output: [1, 1, 128, 320] float32 heatmap (sigmoid activated)
 */
class BallTrackingDetector(context: Context) {

    companion object {
        private const val MODEL_FILE = "ball_tracking.onnx"
        private const val INPUT_W = 320
        private const val INPUT_H = 128
        private const val NUM_PIXELS = INPUT_W * INPUT_H
        private const val CONFIDENCE_THRESHOLD = 0.3f
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    // Circular buffer for 3 consecutive frames
    private val frameBuffer = arrayOfNulls<FloatArray>(3)
    private var frameCount = 0

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

    /**
     * Ball detection result.
     */
    data class BallPosition(
        val x: Float,           // Pixel x in original image coordinates
        val y: Float,           // Pixel y in original image coordinates
        val confidence: Float,  // Peak heatmap value (0-1)
        val detected: Boolean   // Whether ball was detected above threshold
    )

    /**
     * Feed a new frame and detect ball position.
     *
     * Requires at least 3 frames to have been fed before detection begins.
     * Returns null if not enough frames accumulated yet.
     *
     * @param bitmap Camera frame (any size, will be resized to 320x128)
     * @param imageWidth Original image width (for coordinate scaling)
     * @param imageHeight Original image height (for coordinate scaling)
     * @return BallPosition or null if fewer than 3 frames received
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): BallPosition? {
        val startTime = System.nanoTime()

        // Preprocess and add to buffer
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_W, INPUT_H, true)
        val frameData = preprocessBitmap(resized)

        // Shift buffer: [1]→[0], [2]→[1], new→[2]
        frameBuffer[0] = frameBuffer[1]
        frameBuffer[1] = frameBuffer[2]
        frameBuffer[2] = frameData
        frameCount++

        // Need at least 3 frames
        if (frameCount < 3 || frameBuffer[0] == null) {
            return null
        }

        // Concatenate 3 frames into [1, 9, H, W] input
        val inputData = FloatArray(9 * NUM_PIXELS)
        for (i in 0..2) {
            System.arraycopy(frameBuffer[i]!!, 0, inputData, i * 3 * NUM_PIXELS, 3 * NUM_PIXELS)
        }

        // Create tensor and run inference
        val shape = longArrayOf(1, 9, INPUT_H.toLong(), INPUT_W.toLong())
        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), shape)
        val results = session.run(mapOf(inputName to inputTensor))

        // Parse heatmap output [1, 1, 128, 320]
        val outputTensor = results[0] as OnnxTensor
        val heatmapBuffer = outputTensor.floatBuffer
        val heatmap = FloatArray(NUM_PIXELS)
        heatmapBuffer.get(heatmap)

        // Find peak position
        val ballPosition = parseHeatmap(heatmap, imageWidth, imageHeight)

        // Cleanup
        outputTensor.close()
        inputTensor.close()
        results.close()

        lastInferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
        return ballPosition
    }

    /**
     * Convert bitmap to NCHW float array.
     * Simple 0-1 normalization (no ImageNet normalization for TrackNet).
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(NUM_PIXELS)
        bitmap.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)

        val data = FloatArray(3 * NUM_PIXELS)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            data[i] = (pixel shr 16 and 0xFF) / 255f                 // R
            data[NUM_PIXELS + i] = (pixel shr 8 and 0xFF) / 255f     // G
            data[2 * NUM_PIXELS + i] = (pixel and 0xFF) / 255f       // B
        }
        return data
    }

    /**
     * Parse heatmap to extract ball position.
     * Uses argmax to find peak, then scales to original image coordinates.
     */
    private fun parseHeatmap(heatmap: FloatArray, imageWidth: Int, imageHeight: Int): BallPosition {
        var maxVal = 0f
        var maxIdx = 0

        for (i in heatmap.indices) {
            if (heatmap[i] > maxVal) {
                maxVal = heatmap[i]
                maxIdx = i
            }
        }

        val heatmapY = maxIdx / INPUT_W
        val heatmapX = maxIdx % INPUT_W

        // Scale to original image coordinates
        val x = heatmapX.toFloat() / INPUT_W * imageWidth
        val y = heatmapY.toFloat() / INPUT_H * imageHeight

        return BallPosition(
            x = x,
            y = y,
            confidence = maxVal,
            detected = maxVal >= CONFIDENCE_THRESHOLD
        )
    }

    /**
     * Reset frame buffer (e.g., when switching cameras or resuming).
     */
    fun reset() {
        frameBuffer.fill(null)
        frameCount = 0
    }

    fun close() {
        session.close()
    }
}
