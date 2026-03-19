package com.shot.detection

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.shot.core.model.Keypoint
import java.nio.FloatBuffer

/**
 * Detects tennis court keypoints using an ONNX Runtime model.
 *
 * Loads the court_keypoint_v2.onnx model from assets and runs inference
 * on camera frames to detect 8 keypoints (points 9-16) on the near court.
 *
 * Input:  NCHW [1, 3, 256, 256] float32, ImageNet normalized
 * Output: [1, 24] float32 — 8x coords, 8y coords, 8 confidence values
 */
class CourtKeypointDetector(context: Context) {

    companion object {
        private const val MODEL_FILE = "court_keypoint_v2.onnx"
        private const val INPUT_SIZE = 256
        private const val NUM_KEYPOINTS = 8
        private const val OUTPUT_SIZE = NUM_KEYPOINTS * 3 // x, y, confidence

        // ImageNet normalization constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Keypoint IDs in model output order
        private val KEYPOINT_IDS = intArrayOf(9, 10, 11, 12, 13, 14, 15, 16)
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        session = ortEnv.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
    }

    /**
     * Get the inference time of the last detection (for debug display).
     */
    var lastInferenceTimeMs: Long = 0
        private set

    /**
     * Detect keypoints from a camera frame bitmap.
     *
     * @param bitmap Camera frame (any size, will be resized to 256x256)
     * @param imageWidth Original image width (for coordinate scaling)
     * @param imageHeight Original image height (for coordinate scaling)
     * @return List of 8 Keypoint objects with pixel coordinates and confidence
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): List<Keypoint> {
        val startTime = System.nanoTime()

        // Preprocess: resize and convert to NCHW float buffer
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputData = preprocessBitmap(resized)

        // Create ONNX tensor with NCHW shape [1, 3, 256, 256]
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), shape)

        // Run inference
        val results = session.run(mapOf(inputName to inputTensor))

        // Parse output
        val outputTensor = results[0] as OnnxTensor
        val outputValues = outputTensor.floatBuffer
        val values = FloatArray(OUTPUT_SIZE)
        outputValues.get(values)

        val keypoints = parseOutput(values, imageWidth, imageHeight)

        // Cleanup
        outputTensor.close()
        inputTensor.close()
        results.close()

        lastInferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
        return keypoints
    }

    /**
     * Convert a bitmap to the model input format.
     * NCHW format (channels first), ImageNet normalized, float32.
     *
     * Layout: all R values for every pixel, then all G, then all B.
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val numPixels = INPUT_SIZE * INPUT_SIZE
        val data = FloatArray(3 * numPixels)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16 and 0xFF) / 255f
            val g = (pixel shr 8 and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            // NCHW: channel 0 (R) at offset 0, channel 1 (G) at numPixels, channel 2 (B) at 2*numPixels
            data[i] = (r - MEAN[0]) / STD[0]
            data[numPixels + i] = (g - MEAN[1]) / STD[1]
            data[2 * numPixels + i] = (b - MEAN[2]) / STD[2]
        }

        return data
    }

    /**
     * Parse model output into Keypoint objects.
     * Output format: [x0..x7, y0..y7, conf0..conf7] (24 float values)
     * x, y are normalized [0, 1], confidence is [0, 1] (sigmoid activated in model)
     */
    private fun parseOutput(values: FloatArray, imageWidth: Int, imageHeight: Int): List<Keypoint> {
        return (0 until NUM_KEYPOINTS).map { i ->
            val x = values[i] * imageWidth                    // x coords at indices 0-7
            val y = values[NUM_KEYPOINTS + i] * imageHeight    // y coords at indices 8-15
            val confidence = values[2 * NUM_KEYPOINTS + i]     // confidence at indices 16-23

            Keypoint(
                id = KEYPOINT_IDS[i],
                x = x,
                y = y,
                confidence = confidence
            )
        }
    }

    fun close() {
        session.close()
    }
}
