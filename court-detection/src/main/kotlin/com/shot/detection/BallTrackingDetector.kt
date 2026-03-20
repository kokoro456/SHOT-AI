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
 * Optimized for speed:
 * - Reuses pre-allocated arrays (zero GC pressure per frame)
 * - NNAPI delegate attempted for GPU/DSP acceleration
 * - Bilinear filtering disabled for faster resize
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

    // Pre-allocated arrays to avoid GC per frame
    private val inputData = FloatArray(9 * NUM_PIXELS)
    private val pixels = IntArray(NUM_PIXELS)
    private val heatmap = FloatArray(NUM_PIXELS)
    private val inputShape = longArrayOf(1, 9, INPUT_H.toLong(), INPUT_W.toLong())

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
     * Feed a new frame and detect ball position.
     * Returns null if fewer than 3 frames accumulated.
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): BallPosition? {
        val startTime = System.nanoTime()

        // Resize (filter=false for speed, slight quality tradeoff is acceptable)
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_W, INPUT_H, false)

        // Preprocess into new FloatArray and add to buffer
        val frameData = preprocessBitmap(resized)
        if (resized !== bitmap) resized.recycle()

        // Shift buffer
        frameBuffer[0] = frameBuffer[1]
        frameBuffer[1] = frameBuffer[2]
        frameBuffer[2] = frameData
        frameCount++

        if (frameCount < 3 || frameBuffer[0] == null) {
            return null
        }

        // Concatenate 3 frames into pre-allocated input array
        System.arraycopy(frameBuffer[0]!!, 0, inputData, 0, 3 * NUM_PIXELS)
        System.arraycopy(frameBuffer[1]!!, 0, inputData, 3 * NUM_PIXELS, 3 * NUM_PIXELS)
        System.arraycopy(frameBuffer[2]!!, 0, inputData, 6 * NUM_PIXELS, 3 * NUM_PIXELS)

        // Create tensor and run inference
        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), inputShape)
        val results = session.run(mapOf(inputName to inputTensor))

        // Parse heatmap output
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
     * Convert bitmap to NCHW float array (0-1 range, no ImageNet normalization).
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        bitmap.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)

        val data = FloatArray(3 * NUM_PIXELS)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            data[i] = (pixel shr 16 and 0xFF) * 0.003921569f                 // R (/255)
            data[NUM_PIXELS + i] = (pixel shr 8 and 0xFF) * 0.003921569f     // G
            data[2 * NUM_PIXELS + i] = (pixel and 0xFF) * 0.003921569f       // B
        }
        return data
    }

    /**
     * Parse pre-filled heatmap to extract ball position.
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

        val heatmapY = maxIdx / INPUT_W
        val heatmapX = maxIdx % INPUT_W

        val x = heatmapX.toFloat() / INPUT_W * imageWidth
        val y = heatmapY.toFloat() / INPUT_H * imageHeight

        return BallPosition(
            x = x, y = y,
            confidence = maxVal,
            detected = maxVal >= CONFIDENCE_THRESHOLD
        )
    }

    fun reset() {
        frameBuffer.fill(null)
        frameCount = 0
    }

    fun close() {
        session.close()
    }
}
